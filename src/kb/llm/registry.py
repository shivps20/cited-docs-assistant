"""The model registry: the catalogue's profiles turned into ready adapters, per role, behind the privacy policy.

    registry = ModelRegistry(load_catalogue())
    planner = registry.for_role("planner")                  # the adapter that splits comparisons
    chain, notice = registry.answer_chain("claude-opus", context)
    # → [claude-opus, its fallbacks …, the local fallback], filtered by the privacy policy

Privacy policy (independent of which model is chosen): an external model may only see a context in
which every source has external_ok. Otherwise every external model is removed from the chain and the
catalogue's local fallback answers, with a notice saying why. Models whose key is not set are skipped
the same way. The chain always ends with the local fallback, so an answer never depends on an
external service being up.

Adapters are created on first use and kept, so a model that is never used is never contacted.
"""

import dataclasses
from collections.abc import Mapping, Sequence

from kb.llm.catalogue import (
    EXTERNAL,
    LOCAL,
    ROLES,
    CatalogueError,
    ModelCatalogue,
    ModelProfile,
    load_catalogue,
)
from kb.llm.providers import OLLAMA, OPENAI, LLMError, LLMProvider, make_provider
from kb.retrieve.assemble import ContextUnit

AUTO = "auto"      # the answer role's model (also what an empty request means)


class ModelRegistry:
    """Adapters for the catalogue's models, created on first use; role lookup, privacy and fallbacks."""

    def __init__(self, catalogue: ModelCatalogue, providers: Mapping[str, LLMProvider] | None = None):
        """Keep the catalogue; `providers` are ready-made adapters by profile name (tests, old callers)."""
        self.catalogue = catalogue
        self._providers: dict[str, LLMProvider] = dict(providers or {})
        self._given = set(self._providers)

    @classmethod
    def load(cls) -> "ModelRegistry":
        """The registry for KB_MODELS_PATH (or the catalogue built from the older settings)."""
        return cls(load_catalogue())

    @classmethod
    def from_providers(cls, providers: Mapping[str, LLMProvider]) -> "ModelRegistry":
        """A registry around ready adapters named 'ollama' (local) and optionally 'openai' (external):
        the shape used before the catalogue existed, kept for tests and simple callers."""
        if OLLAMA not in providers:
            raise ValueError("from_providers needs a local 'ollama' provider")
        models = {OLLAMA: ModelProfile(OLLAMA, "ollama", providers[OLLAMA].model, location=LOCAL)}
        for name, provider in providers.items():
            if name != OLLAMA:
                models[name] = ModelProfile(name, "openai_compatible", provider.model, location=EXTERNAL,
                                            api_key_env="UNUSED_KEY", fallback=OLLAMA)
        roles = {role: OLLAMA for role in ROLES}
        roles["answer"] = OPENAI if OPENAI in providers else OLLAMA
        return cls(ModelCatalogue(models, roles, OLLAMA, "providers"), providers)

    def provider(self, name: str) -> LLMProvider:
        """The adapter for profile `name`, created on first use."""
        if name not in self._providers:
            self._providers[name] = make_provider(self.catalogue.profile(name))
        return self._providers[name]

    def ready(self, name: str) -> bool:
        """Can profile `name` be called (an adapter was given, or its key is set)?"""
        return name in self._given or self.catalogue.profile(name).ready

    def for_role(self, role: str, *, max_output_tokens: int | None = None) -> LLMProvider:
        """The adapter for `role` (planner, condenser, judge, answer); the local fallback when the role's
        model has no key. max_output_tokens: a separate adapter with a smaller answer cap (condenser)."""
        profile = self.catalogue.for_role(role)
        if not self.ready(profile.name):
            profile = self.catalogue.fallback_profile
        if max_output_tokens is None or profile.name in self._given:
            return self.provider(profile.name)
        capped = dataclasses.replace(profile, max_output_tokens=min(max_output_tokens, profile.max_output_tokens))
        return make_provider(capped)

    def resolve(self, requested: str | None) -> str:
        """The profile name for a request: a model name, 'auto' / empty (the answer role), or the older
        provider names 'ollama' (the local fallback) and 'openai' (a profile named openai)."""
        if not requested or requested == AUTO:
            return self.catalogue.roles["answer"]
        if requested == OLLAMA and OLLAMA not in self.catalogue.models:
            return self.catalogue.fallback
        if requested not in self.catalogue.models:
            raise LLMError(f"unknown model {requested!r}; configured: {', '.join(sorted(self.catalogue.models))}")
        return requested

    def answer_chain(self, requested: str | None, context: Sequence[ContextUnit]) -> tuple[list[ModelProfile], str | None]:
        """The models to try for an answer, in order, and a notice for the user (or None).

        Starts with the requested model and its fallbacks; drops external models when a source may not
        leave the machine, and models without a key; always ends with the local fallback.
        """
        name = self.resolve(requested)
        blocked = sorted({u.doc_id for u in context if not u.external_ok})
        chain, notes = [], []
        for profile in self.catalogue.fallback_chain(name):
            if profile.external and blocked:
                if not notes:
                    notes.append(f"Answered with a local model: {', '.join(blocked)} may not be sent to an "
                                 f"external model such as {profile.name} (external_ok = false).")
                continue
            if not self.ready(profile.name):
                notes.append(f"{profile.name} has no key ({profile.api_key_env} is not set).")
                continue
            chain.append(profile)
        fallback = self.catalogue.fallback_profile
        if fallback not in chain:
            chain.append(fallback)
        notice = " ".join(notes) if chain[0].name != name else None
        return chain, notice

    def describe(self) -> dict:
        """Catalogue source, role → model and fallback, for reports and traces."""
        return {"catalogue": self.catalogue.source, "roles": dict(self.catalogue.roles),
                "fallback": self.catalogue.fallback}


def registry_from(models: "ModelRegistry | Mapping[str, LLMProvider] | None") -> ModelRegistry:
    """A registry from a registry, a mapping of ready adapters, or None (the configured catalogue)."""
    if isinstance(models, ModelRegistry):
        return models
    if models is None:
        return ModelRegistry.load()
    try:
        return ModelRegistry.from_providers(models)
    except ValueError as e:
        raise CatalogueError([str(e)]) from e
