import pytest

from kb.core.config import Settings
from kb.llm.catalogue import CatalogueError, default_catalogue, load_catalogue

GOOD = """
roles:
  answer: claude
  planner: local
fallback: local
models:
  local:
    adapter: ollama
    model: qwen2.5:7b-instruct
    temperature: 0
  mistral:
    adapter: openai_compatible
    base_url: https://api.mistral.ai/v1
    model: mistral-large-latest
    location: external
    api_key_env: MISTRAL_API_KEY
    fallback: local
  claude:
    adapter: anthropic
    model: claude-opus-5-5
    location: external
    api_key_env: ANTHROPIC_API_KEY
    context_tokens: 200000
    max_output_tokens: 4000
    effort: medium
    refusal_retry: false
    fallback: mistral
"""


def settings(**overrides) -> Settings:
    """Settings that ignore the developer's .env (only the given values)."""
    return Settings(_env_file=None, **overrides)


def write(tmp_path, text):
    """A models.yaml with `text`."""
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_catalogue_reads_profiles_roles_and_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    cat = load_catalogue(write(tmp_path, GOOD), settings(ollama_host="http://ollama:11434"))
    assert cat.for_role("answer").name == "claude" and cat.for_role("planner").name == "local"
    assert cat.for_role("condenser").name == "local" and cat.for_role("judge").name == "local"   # missing → fallback
    local, claude = cat.profile("local"), cat.profile("claude")
    assert (local.location, local.base_url, local.temperature, local.external) == ("local", "http://ollama:11434", 0.0, False)
    assert (claude.effort, claude.context_tokens, claude.refusal_retry, claude.external) == ("medium", 200000, False, True)
    assert claude.ready and not cat.profile("mistral").ready                 # key set / not set
    assert [p.name for p in cat.fallback_chain("claude")] == ["claude", "mistral", "local"]
    assert cat.fallback_profile.name == "local" and cat.warnings == ()
    with pytest.raises(CatalogueError, match="unknown model 'gpt'"):
        cat.profile("gpt")


def test_catalogue_lists_every_problem_at_once(tmp_path):
    bad = """
roles: {answer: missing, reviewer: local}
fallback: claude
models:
  local: {adapter: ollama, model: qwen, colour: blue}
  claude: {adapter: anthropic, model: claude-opus-5-5, location: local, effort: huge, api_key_env: anthropic-key}
  gpt: {adapter: openai, model: ""}
  big: {adapter: ollama, model: m, context_tokens: 500, max_output_tokens: 600, temperature: 3}
  loop-a: {adapter: ollama, model: m, fallback: loop-b}
  loop-b: {adapter: ollama, model: m, fallback: loop-a}
  "Bad Name": {adapter: ollama, model: m}
"""
    with pytest.raises(CatalogueError) as err:
        load_catalogue(write(tmp_path, bad), settings())
    text = "\n".join(err.value.errors)
    for expected in ["unknown field(s) colour", "location must be 'external'", "effort must be one of",
                     "api_key_env must be an environment variable name", "adapter must be one of",
                     "'model' (the provider's model id) is required", "context_tokens must be at least",
                     "max_output_tokens must be smaller", "temperature must be a number", "fallbacks loop",
                     "names are lowercase", "unknown role 'reviewer'", "role 'answer': model 'missing'",
                     "fallback 'claude' is not one of the models"]:
        assert expected in text, expected


def test_external_models_need_a_key_variable_and_a_local_fallback(tmp_path):
    text = """
fallback: gpt
models:
  gpt: {adapter: openai_compatible, model: gpt-6.1-sol, location: external}
"""
    with pytest.raises(CatalogueError) as err:
        load_catalogue(write(tmp_path, text), settings())
    assert any("external models need api_key_env" in e for e in err.value.errors)
    ok_key = text.replace("location: external}", "location: external, api_key_env: OPENAI_API_KEY}")
    with pytest.raises(CatalogueError, match="must be a local model"):
        load_catalogue(write(tmp_path, ok_key), settings())


def test_external_planner_or_condenser_gives_a_warning(tmp_path):
    text = GOOD.replace("planner: local", "planner: mistral")
    cat = load_catalogue(write(tmp_path, text), settings())
    assert len(cat.warnings) == 1 and "role 'planner' uses the external model 'mistral'" in cat.warnings[0]


def test_without_a_file_the_catalogue_matches_the_old_settings(tmp_path):
    missing = tmp_path / "none.yaml"
    local_only = load_catalogue(missing, settings(llm_model="qwen2.5:7b-instruct", llm_num_ctx=8192))
    assert list(local_only.models) == ["local"] and set(local_only.roles.values()) == {"local"}
    assert local_only.source == "settings" and local_only.profile("local").temperature == 0.0
    with_openai = default_catalogue(settings(openai_api_key="k", openai_model="gpt-x", KB_LLM_PROVIDER="auto"))
    assert with_openai.roles["answer"] == "openai" and with_openai.roles["planner"] == "local"
    assert with_openai.profile("openai").external and with_openai.profile("openai").fallback == "local"
    forced_local = default_catalogue(settings(openai_api_key="k", openai_model="gpt-x", KB_LLM_PROVIDER="ollama"))
    assert forced_local.roles["answer"] == "local"


def test_the_committed_example_is_valid():
    from kb.core.config import ROOT
    cat = load_catalogue(ROOT / "config" / "models.example.yaml", settings())
    assert {p.adapter for p in cat.models.values()} == {"ollama", "openai_compatible", "anthropic"}
    assert not cat.fallback_profile.external and cat.warnings == ()
