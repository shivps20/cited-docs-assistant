"""Answer pipeline: route -> retrieve -> gate -> prompt -> LLM -> check citations, in one trace.

    answerer = Answerer(conn, retriever, ModelRegistry.load(), not_found_score=0.1)
    answer = answerer.answer(SearchRequest("How do I ...?"), on_token=print)
    print(answer.formatted())

Comparison questions (kb.agent.route) are split into sides by the local LLM and searched once per
side (kb.agent.compare), so every side reaches the context; everything after retrieval is shared.
Which model writes the answer comes from the model registry (kb.llm.registry): the requested model,
its fallbacks and the local fallback, filtered by the privacy policy (external models only see
contexts whose sources all have external_ok).
With cache=True, a question asked again under the same conditions is answered from the answer cache
(kb.answer.cache) without search or the LLM; clean answers are stored there.
"""

import dataclasses
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
from kb.answer.cache import AnswerCache, CachedAnswer, CacheKey, cache_key
from kb.core.tracing import Tracer
from kb.llm.prompts import (
    NOT_FOUND,
    REFERENCES_HEADING,
    build_messages,
    check_citations,
    infer_sources,
    missing_references,
    source_line,
    system_prompt,
)
from kb.llm.providers import Generation, LLMError, LLMProvider
from kb.llm.registry import AUTO, ModelRegistry, registry_from
from kb.retrieve.assemble import ContextUnit, budget_for, fit_context
from kb.retrieve.gate import GateDecision, gate
from kb.retrieve.pipeline import Retriever, SearchRequest, stage_timings
from kb.retrieve.search import Candidate

ANSWERED = "answered"
NOT_FOUND_STATUS = "not_found"
RETRY_UNITS = 3            # refusal retry: the best units of a normal question
RETRY_UNITS_PER_SIDE = 2   # ... and of each side of a comparison
CACHE_VERSION = 1          # bump to retire every cached answer after a change the key does not see


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
    model_profile: str = ""                 # the catalogue model that wrote the answer
    fallback_from: list[str] = field(default_factory=list)   # models that failed before model_profile answered
    cached_from: str = ""                   # cache hit: the trace of the answer that was stored

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

    def __init__(self, conn: sqlite3.Connection, retriever: Retriever,
                 models: ModelRegistry | Mapping[str, LLMProvider] | None = None, *,
                 not_found_score: float = 0.1, provider: str = AUTO, compare: bool = True,
                 planner: LLMProvider | None = None, refusal_retry: bool = False, cache: bool = False):
        """Keep the retriever and the model registry.

        models: a ModelRegistry, a mapping of ready adapters ('ollama', optionally 'openai'), or None
            (the configured catalogue, KB_MODELS_PATH).
        not_found_score: gate threshold on the top rerank score.
        provider: the answer model: a catalogue name, 'auto' (the answer role) or the older 'ollama' /
            'openai'; a question can override it (answer(provider=…)).
        compare: route comparison questions to the comparison path (False: one search for every question).
        planner: the model that splits comparisons into sides (default: the catalogue's planner role).
        refusal_retry: when the LLM refuses, ask once more with only the best-matching units (TD-23).
        cache: answer repeated questions from the answer cache and store clean answers there
            (kb.answer.cache); off for evaluation, which must measure the pipeline itself.
        The read step and the refusal retry are switched per model in the catalogue (compare_read,
        refusal_retry); refusal_retry here can switch the retry off for every model.
        """
        self.conn = conn
        self.retriever = retriever
        self.models = registry_from(models)
        self.not_found_score = not_found_score
        self.provider = provider
        self.compare = compare
        self.planner = planner if planner is not None else self.models.for_role("planner")
        self.refusal_retry = refusal_retry
        self.cache = cache

    def answer(self, req: SearchRequest, *, provider: str | None = None,
               on_token: Callable[[str], None] | None = None,
               on_context: Callable[[list[ContextUnit]], None] | None = None,
               asked: str | None = None, pre_stages: Sequence[tuple[str, float, dict]] = (),
               on_status: Callable[[str, dict], None] | None = None, use_cache: bool = True) -> Answer:
        """on_token receives the answer as it streams; on_context the context, before generation;
        on_status(stage, data) the comparison progress: 'comparing' before the question is split,
        then 'searching_side' {side, index, total} before each side's search, 'reading' {sections}
        when the read step adds sections; 'retrying' {units}
        when a refusal is asked again with fewer sources (on_context then gets the smaller context).

        req.query is what is searched and answered. asked: the user's own words when req.query is a
        condensed follow-up (the trace keeps both). pre_stages: (name, ms, data) of stages timed before
        the trace started, recorded first (e.g. condensing). use_cache=False: answer afresh (the
        answer is still stored when the cache is on).
        """
        requested = provider or self.provider
        target = self.models.catalogue.profile(self.models.resolve(requested))   # the model retrieval is sized for
        budget = budget_for(target.context_tokens, target.max_output_tokens)
        req = dataclasses.replace(req, max_context_units=budget.max_units, max_context_tokens=budget.max_tokens)
        with Tracer(self.conn, asked or req.query, user_id=req.user_id, session_id=req.session_id) as trace:
            if asked and asked != req.query:
                trace.set(standalone_query=req.query)
            for name, ms, data in pre_stages:
                trace.add_stage(name, ms, **data)
            key = self._cache_key(req, target) if self.cache else None
            answer = self._from_cache(key, req, trace, on_token, on_context) if key and use_cache else None
            if answer is None:
                answer = self._answer_fresh(req, trace, requested, target, budget, on_token, on_context, on_status)
                if key and self._cacheable(answer, target):
                    AnswerCache(self.conn).put(key, self._payload(answer))
            trace.set(answer=answer.text, sources=[s.__dict__ for s in answer.sources])
        answer.timings_ms = stage_timings(trace)
        return answer

    def _answer_fresh(self, req: SearchRequest, trace: Tracer, requested: str, target, budget,
                      on_token: Callable[[str], None] | None,
                      on_context: Callable[[list[ContextUnit]], None] | None,
                      on_status: Callable[[str, dict], None] | None) -> Answer:
        """Route, retrieve, gate and generate: the answer when it does not come from the cache."""
        route, plan = self._route(req.query, trace, on_status)
        side_docs: dict[str, str] = {}
        tools = KBTools(self.conn, self.retriever, req, trace)
        if plan is not None and plan.sides:
            total = len(plan.sides)
            on_side = (lambda i, side: on_status("searching_side", {"side": side.label, "index": i, "total": total})
                       ) if on_status else None
            result = retrieve_sides(tools, req.query, plan.sides, on_side=on_side, side_docs=side_docs,
                                    per_side=budget.per_side, max_tokens=budget.compare_tokens)
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
        # Read step (TO-5.10): only for an answer model whose profile has compare_read: true; off for
        # every model by default (with qwen2.5 7B it did not improve answers).
        if sides and target.compare_read and decision.passed:
            self._read_more(answer, tools, side_docs, trace, on_status)
        if on_context:
            on_context(answer.context)
        if decision.passed:
            self._generate(answer, trace, requested, on_token, on_context)
            if (self.refusal_retry and answer.status == NOT_FOUND_STATUS
                    and self._profile_retries(answer.model_profile)):
                self._retry_smaller(answer, trace, requested, on_token, on_context, on_status)
        return answer

    def _cache_key(self, req: SearchRequest, target) -> CacheKey:
        """The cache key of this request for the target answer model (kb.answer.cache)."""
        retrieval = {k: getattr(req, k) for k in ("category", "mode", "rerank", "candidates", "rerank_top",
                                                  "min_context_score", "max_context_units", "max_context_tokens")}
        settings = {"version": CACHE_VERSION, "profile": dataclasses.asdict(target), "prompt": system_prompt(),
                    "compare": self.compare, "refusal_retry": self.refusal_retry,
                    "not_found_score": self.not_found_score, "planner": getattr(self.planner, "model", None)}
        release = f"R{req.release}x" if req.release is not None else None
        return cache_key(self.conn, req.query, groups=req.groups, release=release, retrieval=retrieval,
                         model=target.name, settings=settings)

    @staticmethod
    def _cacheable(answer: Answer, target) -> bool:
        """Store only clean answers: answered, by the requested model itself (no fallback after an error,
        no model swapped by the privacy policy)."""
        return (answer.status == ANSWERED and answer.generation is not None and not answer.fallback_from
                and answer.model_profile == target.name)

    @staticmethod
    def _payload(answer: Answer) -> dict:
        """The answer as stored in the cache (everything but candidates and timings)."""
        return {"trace_id": answer.trace_id, "question": answer.question, "text": answer.text,
                "status": answer.status, "gate": dataclasses.asdict(answer.gate),
                "sources": [dataclasses.asdict(s) for s in answer.sources],
                "context": [dataclasses.asdict(u) for u in answer.context],
                "references": answer.references, "notices": answer.notices,
                "invalid_citations": answer.invalid_citations, "route": answer.route, "sides": answer.sides,
                "retried_from": answer.retried_from, "read_sections": answer.read_sections,
                "model_profile": answer.model_profile,
                "generation": {"provider": answer.generation.provider, "model": answer.generation.model}}

    def _from_cache(self, key: CacheKey, req: SearchRequest, trace: Tracer,
                    on_token: Callable[[str], None] | None,
                    on_context: Callable[[list[ContextUnit]], None] | None) -> Answer | None:
        """The cached answer for `key` as a new Answer in this trace, or None on a miss."""
        with trace.stage("cache", question=key.question, corpus=key.corpus, model=key.model) as stage:
            hit: CachedAnswer | None = AnswerCache(self.conn).get(key)
            stage.update(hit=hit is not None, served_from=hit.trace_id if hit else None)
        if hit is None:
            return None
        p = hit.payload
        answer = Answer(question=req.query, text=p["text"], status=p["status"], gate=hit.gate(),
                        sources=[Source(**s) for s in p["sources"]], context=hit.context(), candidates=[],
                        trace_id=trace.trace_id, timings_ms={}, generation=hit.generation(),
                        invalid_citations=p["invalid_citations"], references=p["references"],
                        notices=[*p["notices"], (f"Answered from the answer cache: the same question was answered "
                                                 f"on {hit.created_at[:16].replace('T', ' ')} UTC (served "
                                                 f"{hit.hit_count}x). A 👎 removes it, so the next ask is answered "
                                                 "afresh.")],
                        route=p["route"], sides=p["sides"], retried_from=p["retried_from"],
                        read_sections=p["read_sections"], model_profile=p["model_profile"], cached_from=hit.trace_id)
        trace.set(cache_hit=True, route=answer.route, gate_decision=answer.gate.decision, index_version=key.corpus,
                  top_rerank_score=answer.gate.top_score, llm_provider=answer.generation.provider,
                  llm_model=answer.generation.model)
        if on_context:
            on_context(answer.context)
        if on_token:
            on_token(answer.text)
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

        Runs only when the answer model's catalogue profile has compare_read: true (off for every model
        by default). Measured with qwen2.5 7B it picked the section holding a missing fact once in 16
        comparisons and the extra text made two answers worse (TO-5.10, TD-14); kept for larger models."""
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
        self._generate(answer, trace, requested, on_token, None)
        if answer.status == ANSWERED:
            answer.retried_from = len(full)
            answer.notices.append(f"Answered on a second attempt with the {len(smaller)} best-matching sources; "
                                  f"with all {len(full)} the model found no answer.")
        else:
            answer.context = full

    def _profile_retries(self, name: str) -> bool:
        """Does the catalogue profile that answered allow the refusal retry (refusal_retry in models.yaml)?"""
        models = self.models.catalogue.models
        return name not in models or models[name].refusal_retry

    def _generate(self, answer: Answer, trace: Tracer, requested: str,
                  on_token: Callable[[str], None] | None,
                  on_context: Callable[[list[ContextUnit]], None] | None = None) -> None:
        """Generate the answer with the first model of the answer chain that works (requested model,
        its fallbacks, the local fallback; external models only for external_ok contexts), then check
        its citations; fills in the answer's text, status and sources.

        Each model gets the context fitted to its own budget (a smaller model than the one retrieval
        was sized for gets the best-ranked part); on_context receives the reduced context when it shrinks.
        """
        chain, notice = self.models.answer_chain(requested, answer.context)
        if notice:
            answer.notices.append(notice)
        full = answer.context

        first = chain[0]
        with trace.stage("generate", profile=first.name, provider=first.name,
                         model=self.models.provider(first.name).model) as stage:
            failed = []
            for i, profile in enumerate(chain):
                context = fit_context(full, budget_for(profile.context_tokens, profile.max_output_tokens),
                                      sides=bool(answer.sides))
                messages = build_messages(answer.question, context, answer.sides)
                stage["prompt_chars"] = sum(len(m["content"]) for m in messages)
                try:
                    generation = self.models.provider(profile.name).generate(messages, on_token=on_token)
                    break
                except LLMError as e:
                    if i == len(chain) - 1:
                        raise
                    # This model is down or refused: answer with the next one rather than not at all.
                    failed.append(profile.name)
                    answer.notices.append(f"{profile.name} failed ({e}); answered with {chain[i + 1].name} instead.")
                    if on_token:
                        on_token("\n")
            if failed:
                stage["fallback_from"] = failed
                answer.fallback_from = failed
            if len(context) < len(full):
                stage["context_fitted"] = [len(full), len(context)]
                answer.notices.append(f"{profile.name} has a smaller context window: answered from the "
                                      f"{len(context)} best of {len(full)} sources.")
                answer.context = context
                if on_context:
                    on_context(context)
            answer.model_profile = profile.name
            stage.update(profile=chain[len(failed)].name, provider=generation.provider, model=generation.model,
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
