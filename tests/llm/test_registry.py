import pytest

from kb.llm.catalogue import ModelCatalogue, ModelProfile
from kb.llm.providers import Generation, LLMError
from kb.llm.registry import ModelRegistry
from kb.retrieve.assemble import ContextUnit


class FakeLLM:
    """A ready adapter that answers with its own name, or fails."""

    def __init__(self, name, fail=False):
        """Name, model and whether generate() raises."""
        self.name, self.model, self.fail, self.calls = name, f"{name}-model", fail, 0

    def generate(self, messages, *, on_token=None, json_format=False, json_schema=None):
        """Answer with the adapter's name (or raise LLMError)."""
        self.calls += 1
        if self.fail:
            raise LLMError(f"{self.name} is down")
        return Generation(f"answer from {self.name}", self.name, self.model, 0.1)


def unit(doc, external_ok):
    """A context unit from document `doc`."""
    return ContextUnit(doc_id=doc, title=doc, section_id=f"{doc}#1", section_number="1", heading_path="1 X",
                       header="h", page_start=1, page_end=1, text="t", kind="section", score=0.9, tokens=10,
                       external_ok=external_ok)


def registry(**fakes):
    """Catalogue: local (fallback), mistral → local, claude → mistral; planner local, answer claude.
    The keyword arguments are ready adapters by profile name; others count as 'no key'."""
    models = {
        "local": ModelProfile("local", "ollama", "qwen"),
        "mistral": ModelProfile("mistral", "openai_compatible", "mistral-small-latest", location="external",
                                api_key_env="REG_TEST_MISTRAL", fallback="local"),
        "claude": ModelProfile("claude", "anthropic", "claude-opus-5-5", location="external",
                               api_key_env="REG_TEST_ANTHROPIC", fallback="mistral"),
    }
    roles = {"answer": "claude", "planner": "local", "condenser": "local", "judge": "local"}
    return ModelRegistry(ModelCatalogue(models, roles, "local", "test"), fakes)


CLEARED = [unit("public-guide", True)]
INTERNAL = [unit("public-guide", True), unit("internal-guide", False)]


def test_answer_chain_follows_fallbacks_and_ends_local():
    reg = registry(local=FakeLLM("local"), mistral=FakeLLM("mistral"), claude=FakeLLM("claude"))
    chain, notice = reg.answer_chain(None, CLEARED)                 # no request: the answer role
    assert [p.name for p in chain] == ["claude", "mistral", "local"] and notice is None
    chain, _ = reg.answer_chain("mistral", CLEARED)
    assert [p.name for p in chain] == ["mistral", "local"]
    assert [p.name for p in reg.answer_chain("local", CLEARED)[0]] == ["local"]


def test_internal_sources_never_reach_an_external_model():
    reg = registry(local=FakeLLM("local"), mistral=FakeLLM("mistral"), claude=FakeLLM("claude"))
    chain, notice = reg.answer_chain("claude", INTERNAL)
    assert [p.name for p in chain] == ["local"]
    assert "internal-guide may not be sent to an external model such as claude" in notice
    assert "public-guide" not in notice


def test_models_without_a_key_are_skipped(monkeypatch):
    monkeypatch.delenv("REG_TEST_ANTHROPIC", raising=False)
    reg = registry(local=FakeLLM("local"), mistral=FakeLLM("mistral"))    # claude: no adapter, no key
    chain, notice = reg.answer_chain(None, CLEARED)
    assert [p.name for p in chain] == ["mistral", "local"] and "claude has no key (REG_TEST_ANTHROPIC" in notice


def test_roles_and_requests():
    reg = registry(local=FakeLLM("local"), claude=FakeLLM("claude"))
    assert reg.for_role("planner").name == "local" and reg.for_role("answer").name == "claude"
    assert reg.resolve("auto") == "claude" and reg.resolve("ollama") == "local"      # older provider names
    with pytest.raises(LLMError, match="unknown model 'gpt'"):
        reg.resolve("gpt")


def test_a_role_whose_model_has_no_key_uses_the_local_fallback(monkeypatch):
    monkeypatch.delenv("REG_TEST_ANTHROPIC", raising=False)
    reg = registry(local=FakeLLM("local"))
    assert reg.for_role("answer").name == "local"


def test_from_providers_keeps_the_older_shape():
    reg = ModelRegistry.from_providers({"ollama": FakeLLM("ollama"), "openai": FakeLLM("openai")})
    assert reg.catalogue.roles["answer"] == "openai" and reg.for_role("planner").name == "ollama"
    assert [p.name for p in reg.answer_chain("auto", INTERNAL)[0]] == ["ollama"]
    assert [p.name for p in reg.answer_chain("openai", CLEARED)[0]] == ["openai", "ollama"]
