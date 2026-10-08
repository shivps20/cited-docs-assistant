"""Answer pipeline: retrieve -> gate -> prompt -> LLM -> check citations, in one trace.

    answerer = Answerer(conn, retriever, build_providers(settings), not_found_score=0.1)
    answer = answerer.answer(SearchRequest("How do I ...?"), on_token=print)
    print(answer.formatted())
"""

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

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
                 not_found_score: float = 0.1, provider: str = "auto"):
        """Keep the retriever and the available LLM providers (by name).

        not_found_score: gate threshold on the top rerank score.
        provider: default provider choice ('auto', 'ollama' or 'openai').
        """
        self.conn = conn
        self.retriever = retriever
        self.providers = providers
        self.not_found_score = not_found_score
        self.provider = provider

    def answer(self, req: SearchRequest, *, provider: str | None = None,
               on_token: Callable[[str], None] | None = None,
               on_context: Callable[[list[ContextUnit]], None] | None = None,
               asked: str | None = None, pre_stages: Sequence[tuple[str, float, dict]] = ()) -> Answer:
        """on_token receives the answer as it streams; on_context the context, before generation.

        req.query is what is searched and answered. asked: the user's own words when req.query is a
        condensed follow-up (the trace keeps both). pre_stages: (name, ms, data) of stages timed before
        the trace started, recorded first (e.g. condensing).
        """
        with Tracer(self.conn, asked or req.query, user_id=req.user_id, session_id=req.session_id) as trace:
            trace.set(route="answer")
            if asked and asked != req.query:
                trace.set(standalone_query=req.query)
            for name, ms, data in pre_stages:
                trace.add_stage(name, ms, **data)
            result = self.retriever.retrieve(req, trace)

            with trace.stage("gate", threshold=self.not_found_score) as stage:
                decision = gate(result, self.not_found_score)
                stage.update(decision=decision.decision, top_score=decision.top_score)
            trace.set(gate_decision=decision.decision)

            answer = Answer(question=req.query, text=NOT_FOUND, status=NOT_FOUND_STATUS, gate=decision,
                            sources=[], context=result.context, candidates=result.candidates,
                            trace_id=trace.trace_id, timings_ms={})
            if on_context:
                on_context(result.context)
            if decision.passed:
                self._generate(answer, trace, provider or self.provider, on_token)
            trace.set(answer=answer.text, sources=[s.__dict__ for s in answer.sources])
        answer.timings_ms = stage_timings(trace)
        return answer

    def _generate(self, answer: Answer, trace: Tracer, requested: str,
                  on_token: Callable[[str], None] | None) -> None:
        """Pick the provider, generate the answer (falling back from OpenAI to Ollama on failure)
        and check its citations; fills in the answer's text, status and sources.
        """
        name, notice = select_provider(requested, answer.context, list(self.providers))
        if notice:
            answer.notices.append(notice)
        messages = build_messages(answer.question, answer.context)

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
