"""The model catalogue: which language models exist and which job each one does (Phase 6).

Models are declared in a local YAML file (`config/models.yaml`, path `KB_MODELS_PATH`, git-ignored;
template `config/models.example.yaml`). Each entry is a *profile* chosen by name:

    adapter       ollama | openai_compatible | anthropic       (how to talk to it)
    model         the provider's model id, e.g. qwen2.5:7b-instruct, claude-opus-5-5
    location      local | external   (external models only see sources with external_ok)
    base_url      server / API address (ollama: OLLAMA_HOST; openai_compatible: OpenAI when empty)
    api_key_env   name of the environment variable holding the key (keys stay in .env)
    context_tokens, max_output_tokens, temperature, effort, timeout, keep_alive, json_mode
    refusal_retry, compare_read      per-model features (TD-23, TO-5.10)
    fallback      the profile to use when this one fails or is not reachable

`roles` says which profile does which job (answer, planner, condenser, judge); `fallback` names the
local profile that answers when an external model may not see the context. Without a file the
catalogue is built from the older settings (LLM_MODEL, OPENAI_*, KB_LLM_PROVIDER), so behaviour is
unchanged until a catalogue is written.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from kb.core.config import Settings, get_settings

ADAPTERS = ("ollama", "openai_compatible", "anthropic")
LOCAL, EXTERNAL = "local", "external"
ROLES = ("answer", "planner", "condenser", "judge")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MIN_CONTEXT_TOKENS = 1024

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ENV = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_FIELDS = {"adapter", "model", "location", "base_url", "api_key_env", "context_tokens", "max_output_tokens",
           "temperature", "effort", "timeout", "keep_alive", "json_mode", "refusal_retry", "compare_read",
           "fallback"}


class CatalogueError(Exception):
    """models.yaml is invalid, or a name does not exist; the message lists every problem."""

    def __init__(self, errors: list[str]):
        """Keep the list of problems and build a message listing them all."""
        super().__init__(f"{len(errors)} models.yaml error(s):\n" + "\n".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class ModelProfile:
    """One configured language model, chosen by name."""

    name: str
    adapter: str
    model: str
    location: str = LOCAL
    base_url: str = ""
    api_key_env: str = ""
    context_tokens: int = 8192
    max_output_tokens: int = 1500
    temperature: float | None = None     # None: the provider's default (some models reject it)
    effort: str = ""                     # anthropic only: low … max
    timeout: float = 600.0
    keep_alive: str = "10m"              # ollama only
    json_mode: bool = True               # the provider can be asked for a JSON-only reply
    refusal_retry: bool = True
    compare_read: bool = False
    fallback: str = ""

    @property
    def external(self) -> bool:
        """Does this model run outside the machine (and so may only see external_ok sources)?"""
        return self.location == EXTERNAL

    @property
    def api_key(self) -> str:
        """The key from the environment (.env is exported at startup); '' when not set."""
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    @property
    def ready(self) -> bool:
        """Is everything present to call it (the key, when the profile names one)?"""
        return not self.api_key_env or bool(self.api_key)


@dataclass(frozen=True)
class ModelCatalogue:
    """All configured models, the profile per role, and the local fallback."""

    models: dict[str, ModelProfile]
    roles: dict[str, str]
    fallback: str
    source: str = "settings"             # the file it was read from, or 'settings'
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def profile(self, name: str) -> ModelProfile:
        """The named profile; CatalogueError when there is none."""
        if name not in self.models:
            raise CatalogueError([f"unknown model {name!r}; configured: {', '.join(sorted(self.models))}"])
        return self.models[name]

    def for_role(self, role: str) -> ModelProfile:
        """The profile that does `role` (answer, planner, condenser, judge)."""
        if role not in ROLES:
            raise CatalogueError([f"unknown role {role!r}; roles: {', '.join(ROLES)}"])
        return self.models[self.roles[role]]

    @property
    def fallback_profile(self) -> ModelProfile:
        """The local model used when an external one may not see the context."""
        return self.models[self.fallback]

    def fallback_chain(self, name: str) -> list[ModelProfile]:
        """`name` followed by its fallbacks in order (validated to end without a loop)."""
        chain, current = [], name
        while current and current not in [p.name for p in chain]:
            chain.append(self.models[current])
            current = self.models[current].fallback
        return chain


def default_catalogue(settings: Settings) -> ModelCatalogue:
    """The catalogue without a models.yaml: the local Ollama model, plus OpenAI when it is configured.

    Mirrors the behaviour before Phase 6: KB_LLM_PROVIDER auto / openai answers with OpenAI when it is
    configured (the privacy check then decides per context); everything else runs locally.
    """
    local = ModelProfile(name="local", adapter="ollama", model=settings.llm_model, location=LOCAL,
                         base_url=settings.ollama_host, context_tokens=settings.llm_num_ctx,
                         max_output_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature,
                         keep_alive=settings.llm_keep_alive, refusal_retry=settings.refusal_retry)
    models = {"local": local}
    answer = "local"
    if settings.openai_api_key and settings.openai_model:
        models["openai"] = ModelProfile(name="openai", adapter="openai_compatible", model=settings.openai_model,
                                        location=EXTERNAL, api_key_env="OPENAI_API_KEY",
                                        max_output_tokens=settings.llm_max_tokens, timeout=120.0,
                                        refusal_retry=settings.refusal_retry, fallback="local")
        if settings.llm_provider != "ollama":
            answer = "openai"
    roles = {role: "local" for role in ROLES}
    roles["answer"] = answer
    return ModelCatalogue(models, roles, "local")


def load_catalogue(path: Path | None = None, settings: Settings | None = None) -> ModelCatalogue:
    """Read and validate models.yaml (default: KB_MODELS_PATH); without the file, the default catalogue.

    Raises CatalogueError listing every problem found.
    """
    settings = settings or get_settings()
    path = path or settings.models_path
    if not path.exists():
        return default_catalogue(settings)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise CatalogueError([f"{path.name} is not valid YAML: {e}"]) from e
    if not isinstance(data, dict):
        raise CatalogueError([f"{path.name}: expected a mapping with 'models', 'roles' and 'fallback'"])

    errors: list[str] = []
    raw_models = data.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise CatalogueError(["'models' must be a mapping with at least one model"])
    models: dict[str, ModelProfile] = {}
    for name, spec in raw_models.items():
        profile = _profile(str(name), spec, settings, errors)
        if profile is not None:
            models[profile.name] = profile

    fallback = str(data.get("fallback") or "")
    if fallback not in models:
        errors.append(f"fallback {fallback!r} is not one of the models")
    elif models[fallback].external:
        errors.append(f"fallback {fallback!r} must be a local model (it answers when an external one may not)")

    roles, warnings = _roles(data.get("roles"), models, fallback, errors)
    _check_fallbacks(models, errors)
    if errors:
        raise CatalogueError(errors)
    return ModelCatalogue(models, roles, fallback, str(path), tuple(warnings))


def _profile(name: str, spec, settings: Settings, errors: list[str]) -> ModelProfile | None:
    """One validated profile, or None (with the problems appended to `errors`)."""
    where = f"model {name!r}"
    if not _NAME.match(name):
        errors.append(f"{where}: names are lowercase letters, digits, '.', '-' or '_'")
        return None
    if not isinstance(spec, dict):
        errors.append(f"{where}: expected a mapping with at least 'adapter' and 'model'")
        return None
    before = len(errors)
    unknown = sorted(set(spec) - _FIELDS)
    if unknown:
        errors.append(f"{where}: unknown field(s) {', '.join(unknown)}")
    adapter = str(spec.get("adapter") or "")
    if adapter not in ADAPTERS:
        errors.append(f"{where}: adapter must be one of {', '.join(ADAPTERS)} (got {adapter!r})")
    model = str(spec.get("model") or "").strip()
    if not model:
        errors.append(f"{where}: 'model' (the provider's model id) is required")
    location = str(spec.get("location") or (LOCAL if adapter == "ollama" else EXTERNAL))
    if location not in (LOCAL, EXTERNAL):
        errors.append(f"{where}: location must be 'local' or 'external' (got {location!r})")
    if adapter == "anthropic" and location != EXTERNAL:
        errors.append(f"{where}: the anthropic adapter calls an external API; location must be 'external'")
    api_key_env = str(spec.get("api_key_env") or "")
    if api_key_env and not _ENV.match(api_key_env):
        errors.append(f"{where}: api_key_env must be an environment variable name like ANTHROPIC_API_KEY")
    if location == EXTERNAL and not api_key_env:
        errors.append(f"{where}: external models need api_key_env (the key itself stays in .env)")
    context = _int(spec, "context_tokens", 8192, where, errors)
    output = _int(spec, "max_output_tokens", 1500, where, errors)
    if context is not None and context < MIN_CONTEXT_TOKENS:
        errors.append(f"{where}: context_tokens must be at least {MIN_CONTEXT_TOKENS}")
    if context is not None and output is not None and output >= context:
        errors.append(f"{where}: max_output_tokens must be smaller than context_tokens")
    temperature = spec.get("temperature")
    if temperature is not None and (not isinstance(temperature, int | float) or not 0 <= temperature <= 2):
        errors.append(f"{where}: temperature must be a number from 0 to 2 (or left out)")
    effort = str(spec.get("effort") or "")
    if effort and adapter != "anthropic":
        errors.append(f"{where}: effort is only used by the anthropic adapter")
    elif effort and effort not in EFFORTS:
        errors.append(f"{where}: effort must be one of {', '.join(EFFORTS)}")
    timeout = spec.get("timeout", 600.0 if adapter == "ollama" else 120.0)
    if not isinstance(timeout, int | float) or timeout <= 0:
        errors.append(f"{where}: timeout must be a positive number of seconds")
    flags = {f: spec.get(f, d) for f, d in (("json_mode", True), ("refusal_retry", True), ("compare_read", False))}
    for flag, value in flags.items():
        if not isinstance(value, bool):
            errors.append(f"{where}: {flag} must be true or false")
    if len(errors) > before:
        return None
    base_url = str(spec.get("base_url") or (settings.ollama_host if adapter == "ollama" else ""))
    return ModelProfile(name=name, adapter=adapter, model=model, location=location, base_url=base_url,
                        api_key_env=api_key_env, context_tokens=context, max_output_tokens=output,
                        temperature=None if temperature is None else float(temperature), effort=effort,
                        timeout=float(timeout), keep_alive=str(spec.get("keep_alive") or "10m"),
                        fallback=str(spec.get("fallback") or ""), **flags)


def _int(spec: dict, key: str, default: int, where: str, errors: list[str]) -> int | None:
    """A positive integer field, or None (with the problem appended)."""
    value = spec.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{where}: {key} must be a positive whole number")
        return None
    return value


def _roles(raw, models: dict[str, ModelProfile], fallback: str, errors: list[str]) -> tuple[dict[str, str], list[str]]:
    """The profile per role (missing roles: the fallback), and warnings about external planner / condenser."""
    raw = raw or {}
    if not isinstance(raw, dict):
        errors.append("'roles' must be a mapping of role → model name")
        raw = {}
    for role in sorted(set(raw) - set(ROLES)):
        errors.append(f"unknown role {role!r}; roles: {', '.join(ROLES)}")
    roles, warnings = {}, []
    for role in ROLES:
        name = str(raw.get(role) or fallback)
        if name not in models:
            errors.append(f"role {role!r}: model {name!r} is not configured")
            continue
        roles[role] = name
        if role in ("planner", "condenser") and models[name].external:
            warnings.append(f"role {role!r} uses the external model {name!r}: questions and chat history "
                            "will be sent to it (they may name internal topics)")
    return roles, warnings


def _check_fallbacks(models: dict[str, ModelProfile], errors: list[str]) -> None:
    """Every fallback names a configured model, and following fallbacks never loops."""
    for profile in models.values():
        if profile.fallback and profile.fallback not in models:
            errors.append(f"model {profile.name!r}: fallback {profile.fallback!r} is not configured")
    for profile in models.values():
        seen, current = [], profile.name
        while current in models and models[current].fallback:
            if current in seen:
                errors.append(f"model {profile.name!r}: fallbacks loop ({' → '.join([*seen, current])})")
                break
            seen.append(current)
            current = models[current].fallback
