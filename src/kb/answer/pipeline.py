"""Answer pipeline: route -> retrieve -> gate -> prompt -> LLM -> check citations, in one trace.

    answerer = Answerer(conn, retriever, build_providers(settings), not_found_score=0.1)
    answer = answerer.answer(SearchRequest("How do I ...?"), on_token=print)
    print(answer.formatted())

Comparison questions (kb.agent.route) are split into sides by the local LLM and searched once per
side (kb.agent.compare), so every side reaches the context; everything after retrieval is shared.
"""

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from kb.agent.compare import (
    Decomposition,
    decompose,
    plan_reads,
    read_sections,
    retrieve_sides,
)
from kb.agent.route import ANSWER, COMPARE, Route, route_question
from kb.agent.tools import KBTools
from kb.core.tracing import Tracer
from kb.llm.prompts import (
    NOT_FOUND,
    REFERENCES_HEADING,
    build_messages,
    check_citations,
    infer_sources,
    missing_references,
    source_line,
)
from kb.llm.providers import (
    OLLAMA,
    OPENAI,
    Generation,
    LLMError,
    LLMProvider,
    select_provider,
)
from kb.retrieve.assemble import ContextUnit
from kb.retrieve.gate import GateDecision, gate
from kb.retrieve.pipeline import Retriever, SearchRequest, stage_timings
from kb.retrieve.search import Candidate

ANSWERED = "answered"
NOT_FOUND_STATUS = "not_found"
RETRY_UNITS = 3            # refusal retry: the best units of a normal question
RETRY_UNITS_PER_SIDE = 2   # ... and of each side of a comparison


@dataclass
class Source:
    """A source cited in an answer: its [n], document, release, section and pages."""
    n: int                      # the [n] used in the answer text
    doc_id: str
    title: str
    release: str
    section: str                # section number, e.g. '2.2.3'
    heading: str
    page_start: int
    page_end: int
    line: str                   # formatted source line
    same_text: list[dict] = field(default_factory=list)   # near-identical copies in other documents:
                                                          # doc_id, title, section, page_start, page_end, release


@dataclass
class Answer:
    """Everything about one answer: text, status, gate decision, sources, context, generation and timings.
    """
    question: str
    text: str
    status: str                             # answered | not_found
    gate: GateDecision
    sources: list[Source]
    context: list[ContextUnit]
    candidates: list[Candidate]
    trace_id: str
    timings_ms: dict[str, float]
    generation: Generation | None = None
    invalid_citations: list[int] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)  # article numbers / URLs added from cited sources
    route: str = ANSWER                     # answer | compare (the path that produced the answer)
    sides: list[str] = field(default_factory=list)       # comparisons: the items compared
    retried_from: int = 0                   # units of the first attempt, when a refusal was retried
    read_sections: int = 0                  # sections added by the comparison read step

    @property
    def refused_by(self) -> str | None:
        """'gate' or 'llm' when not found, else None."""
        if self.status != NOT_FOUND_STATUS:
            return None
        return "llm" if self.generation else "gate"

    def formatted(self) -> str:
        """The answer text followed by its Sources list, ready to print."""
        lines = [self.text]
        if self.sources:
            lines += ["", "Sources:"] + [f"  {s.line}" for s in self.sources]
        return "\n".join(lines)


def retry_context(context: Sequence[ContextUnit], sides: Sequence[str]) -> list[ContextUnit]:
    """The smaller context for a second attempt after a refusal: the first RETRY_UNITS units (the
    context is in rerank order), or for a comparison the first RETRY_UNITS_PER_SIDE units of each side."""
    if not sides:
        return list(context[:RETRY_UNITS])
    taken: dict[str, int] = {}
    smaller = []
    for unit in context:
        if taken.get(unit.side, 0) < RETRY_UNITS_PER_SIDE:
            taken[unit.side] = taken.get(unit.side, 0) + 1
            smaller.append(unit)
    return smaller


def _source(n: int, unit: ContextUnit) -> Source:
    """Source entry for context unit `unit`, cited as [n]."""
    return Source(n=n, doc_id=unit.doc_id, title=unit.title, release=unit.release, section=unit.section_number,
                  heading=unit.heading, page_start=unit.page_start, page_end=unit.page_end,
                  line=source_line(n, unit),
                  same_text=[{"doc_id": s.doc_id, "title": s.title, "section": s.section_number,
                              "page_start": s.page_start, "page_end": s.page_end, "release": s.release}
                             for s in unit.same_text])


class Answerer:
    """Answers questions: retrieval, confidence gate, LLM generation and citation checking, in one trace."""

    def __init__(self, conn: sqlite3.Connection, retriever: Retriever, providers: Mapping[str, LLMProvider], *,
                 not_found_score: float = 0.1, provider: str = "auto", compare: bool = True,
                 planner: LLMProvider | None = None, refusal_retry: bool = False):
        # Read step disabled (TO-5.1, TD-14): with qwen2.5 7B it did not improve answers; re-enable with a larger model.
        # To re-enable, add the parameter back:  compare_read: bool = False
        """Keep the retriever and the available LLM providers (by name).

        not_found_score: gate threshold on the top rerank score.
        provider: default provider choice ('auto', 'ollama' or 'openai').
        compare: route comparison questions to the comparison path (False: one search for every question).
        planner: local model that splits comparisons into sides (default: the Ollama answer provider).
        refusal_retry: when the LLM refuses, ask once more with only the best-matching units (TD-23).
        (compare_read, disabled: after a comparison's per-side searches, let the local LLM pick more
            sections of each side's guide from its table of contents; see _read_more.)
        """
        self.conn = conn
        self.retriever = retriever
        self.providers = providers
        self.not_found_score = not_found_score
        self.provider = provider
        self.compare = compare
        self.planner = planner if planner is not None else providers.get(OLLAMA)
        self.refusal_retry = refusal_retry
        # self.compare_read = compare_read      # read step disabled (TO-5.1)

    def answer(self, req: SearchRequest, *, provider: str | None = None,
               on_token: Callable[[str], None] | None = None,
               on_context: Callable[[list[ContextUnit]], None] | None = None,
               asked: str | None = None, pre_stages: Sequence[tuple[str, float, dict]] = (),
               on_status: Callable[[str, dict], None] | None = None) -> Answer:
        """on_token receives the answer as it streams; on_context the context, before generation;
        on_status(stage, data) the comparison progress: 'comparing' before the question is split,
        then 'searching_side' {side, index, total} before each side's search, 'reading' {sections}
        when the read step adds sections; 'retrying' {units}
        when a refusal is asked again with fewer sources (on_context then gets the smaller context).

        req.query is what is searched and answered. asked: the user's own words when req.query is a
        condensed follow-up (the trace keeps both). pre_stages: (name, ms, data) of stages timed before
        the trace started, recorded first (e.g. condensing).
        """
        with Tracer(self.conn, asked or req.query, user_id=req.user_id, session_id=req.session_id) as trace:
            if asked and asked != req.query:
                trace.set(standalone_query=req.query)
            for name, ms, data in pre_stages:
                trace.add_stage(name, ms, **data)
            route, plan = self._route(req.query, trace, on_status)
            side_docs: dict[str, str] = {}
            tools = KBTools(self.conn, self.retriever, req, trace)
            if plan is not None and plan.sides:
                total = len(plan.sides)
                on_side = (lambda i, side: on_status("searching_side", {"side": side.label, "index": i, "total": total})
                           ) if on_status else None
                result = retrieve_sides(tools, req.query, plan.sides, on_side=on_side, side_docs=side_docs)
                trace.set(top_rerank_score=result.candidates[0].rerank_score if result.candidates else None)
            else:
                result = self.retriever.retrieve(req, trace)
            sides = [s.label for s in plan.sides] if plan is not None else []
            trace.set(route=COMPARE if sides else ANSWER)

            with trace.stage("gate", threshold=self.not_found_score) as stage:
                decision = gate(result, self.not_found_score)
                stage.update(decision=decision.decision, top_score=decision.top_score)
            trace.set(gate_decision=decision.decision)

            answer = Answer(question=req.query, text=NOT_FOUND, status=NOT_FOUND_STATUS, gate=decision,
                            sources=[], context=result.context, candidates=result.candidates,
                            trace_id=trace.trace_id, timings_ms={}, route=COMPARE if sides else ANSWER,
                            sides=sides)
            if route.kind == COMPARE and not sides:
                answer.notices.append(f"Comparison answered with one search ({plan.reason if plan else route.reason}).")
            # Read step disabled (TO-5.1, TD-14): with qwen2.5 7B it did not improve answers; re-enable with a larger model.
            # if sides and self.compare_read and decision.passed:
            #     self._read_more(answer, tools, side_docs, trace, on_status)
            if on_context:
                on_context(result.context)
            if decision.passed:
                self._generate(answer, trace, provider or self.provider, on_token)
                if self.refusal_retry and answer.status == NOT_FOUND_STATUS:
                    self._retry_smaller(answer, trace, provider or self.provider, on_token, on_context, on_status)
            trace.set(answer=answer.text, sources=[s.__dict__ for s in answer.sources])
        answer.timings_ms = stage_timings(trace)
        return answer

    def _route(self, question: str, trace: Tracer,
               on_status: Callable[[str, dict], None] | None = None) -> tuple[Route, Decomposition | None]:
        """Route the question; for a comparison, split it into sides (None when not a comparison)."""
        with trace.stage("route") as stage:
            route = route_question(question)
            if route.kind == COMPARE and not self.compare:
                route = Route(ANSWER, "comparison path switched off")
            stage.update(route=route.kind, reason=route.reason)
        if route.kind != COMPARE:
            return route, None
        if on_status:
            on_status("comparing", {})
        plan = decompose(self.planner, question)
        trace.add_stage("decompose", plan.seconds * 1000, reason=plan.reason,
                        sides=[{"label": s.label, "query": s.query} for s in plan.sides], raw_output=plan.raw)
        return route, plan

    def _read_more(self, answer: Answer, tools: KBTools, side_docs: dict[str, str], trace: Tracer,
                   on_status: Callable[[str, dict], None] | None) -> None:
        """The comparison read step: the local LLM picks more sections of each side's guide from its
        table of contents; they are read (access-checked) and appended to the context.

        Not called at the moment (see the commented call in answer(), TO-5.1 / TD-14): measured with
        qwen2.5 7B it picked the section holding a missing fact once in 16 comparisons, and the extra
        text made two answers worse. Kept for a larger model."""
        reads = plan_reads(self.planner, tools, answer.question, answer.context, side_docs)
        units = read_sections(tools, reads.reads)
        trace.add_stage("read", reads.seconds * 1000, reason=reads.reason, chosen=[sid for _, sid in reads.reads],
                        added=[[u.section_id, u.side, u.tokens] for u in units], raw_output=reads.raw)
        if units:
            if on_status:
                on_status("reading", {"sections": [u.citation for u in units]})
            answer.context = [*answer.context, *units]
            answer.read_sections = len(units)

    def _retry_smaller(self, answer: Answer, trace: Tracer, requested: str,
                       on_token: Callable[[str], None] | None,
                       on_context: Callable[[list[ContextUnit]], None] | None,
                       on_status: Callable[[str, dict], None] | None) -> None:
        """Second attempt after an LLM refusal, with only the best-matching units (TD-23).

        Keeps the refusal (and the full context) when the smaller context is not smaller or the
        model refuses again; unanswerable questions are expected to be refused twice.
        """
        full = answer.context
        smaller = retry_context(full, answer.sides)
        if not smaller or len(smaller) >= len(full):
            return
        trace.add_stage("retry", 0.0, units=[u.section_id for u in smaller], first_units=len(full))
        if on_status:
            on_status("retrying", {"units": len(smaller)})
        if on_context:
            on_context(smaller)
        if on_token:
            on_token("\n\n")
        answer.context = smaller
        self._generate(answer, trace, requested, on_token)
        if answer.status == ANSWERED:
            answer.retried_from = len(full)
            answer.notices.append(f"Answered on a second attempt with the {len(smaller)} best-matching sources; "
                                  f"with all {len(full)} the model found no answer.")
        else:
            answer.context = full

    def _generate(self, answer: Answer, trace: Tracer, requested: str,
                  on_token: Callable[[str], None] | None) -> None:
        """Pick the provider, generate the answer (falling back from OpenAI to Ollama on failure)
        and check its citations; fills in the answer's text, status and sources.
        """
        name, notice = select_provider(requested, answer.context, list(self.providers))
        if notice:
            answer.notices.append(notice)
        messages = build_messages(answer.question, answer.context, answer.sides)

        with trace.stage("generate", provider=name, model=self.providers[name].model,
                         prompt_chars=sum(len(m["content"]) for m in messages)) as stage:
            try:
                generation = self.providers[name].generate(messages, on_token=on_token)
            except LLMError as e:
                if name != OPENAI:
                    raise
                # External provider down: answer locally rather than not at all.
                answer.notices.append(f"OpenAI failed ({e}); answered with the local model instead.")
                stage["fallback_from"] = OPENAI
                if on_token:
                    on_token("\n")
                generation = self.providers[OLLAMA].generate(messages, on_token=on_token)
            stage.update(provider=generation.provider, model=generation.model,
                         prompt_tokens=generation.prompt_tokens, output_tokens=generation.output_tokens,
                         load_s=generation.load_seconds, tokens_per_s=generation.tokens_per_s,
                         raw_output=generation.text)   # before citation clean-up, for debugging
        answer.generation = generation
        trace.set(llm_provider=generation.provider, llm_model=generation.model)

        with trace.stage("cite") as stage:
            citations = check_citations(generation.text, len(answer.context))
            stage.update(cited=citations.cited, invalid=citations.invalid, refused=citations.refused)
        answer.text = citations.text
        answer.invalid_citations = citations.invalid
        if citations.refused:
            return
        answer.status = ANSWERED
        if citations.dropped_not_found:
            answer.notices.append("Removed the contradictory 'could not find the answer' sentence the model "
                                  "added to its answer.")
        numbers = citations.cited
        if not numbers:
            numbers = infer_sources(answer.text, answer.context)
            answer.notices.append(
                "The model wrote no [n] markers; the sources above were identified from the commands it quotes."
                if numbers else "The model wrote no [n] markers and no source could be identified; "
                                "check the answer against the context (--show-context).")
        answer.sources = [_source(n, answer.context[n - 1]) for n in sorted(numbers)]
        answer.references = missing_references(answer.text, [(n, answer.context[n - 1]) for n in sorted(numbers)])
        if answer.references:   # part of the stored answer, so traces and evaluation see it too
            answer.text = "\n".join([answer.text, "", REFERENCES_HEADING, *answer.references])
