from types import SimpleNamespace

import openai
import pytest

from kb.llm.catalogue import ModelProfile
from kb.llm.providers import (
    JSON_ONLY,
    AnthropicProvider,
    LLMError,
    OllamaProvider,
    OpenAIProvider,
    make_provider,
    strip_json_fences,
)

MESSAGES = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Question?"}]


class FakeCompletions:
    """Records create() calls and streams fixed chunks the way the openai SDK does."""

    def __init__(self, pieces, error=None):
        """Pieces to stream, or an exception to raise."""
        self.pieces, self.error, self.calls = pieces, error, []

    def create(self, **kwargs):
        """One streamed completion: a chunk per piece, then a usage-only chunk."""
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))], usage=None)
                  for p in self.pieces]
        return iter([*chunks, SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3))])


def openai_provider(pieces, error=None, **kwargs):
    """An OpenAIProvider whose client is a fake."""
    provider = OpenAIProvider("key", "mistral-medium-latest", base_url="https://api.mistral.ai/v1",
                              name="mistral", **kwargs)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(pieces, error)))
    return provider


def calls(provider):
    """The create() calls the fake client received."""
    return provider.client.chat.completions.calls


def test_openai_compatible_streams_and_reports_usage():
    seen = []
    provider = openai_provider(["Hel", "lo"])
    g = provider.generate(MESSAGES, on_token=seen.append)
    assert (g.text, g.provider, g.model, g.prompt_tokens, g.output_tokens) == ("Hello", "mistral", "mistral-medium-latest", 11, 3)
    assert seen == ["Hel", "lo"]
    sent = calls(provider)[0]
    assert sent["messages"] == MESSAGES and "temperature" not in sent and "response_format" not in sent
    assert provider.client is not None and provider.base_url == "https://api.mistral.ai/v1"


def test_openai_compatible_json_mode_and_temperature():
    provider = openai_provider(['{"ok": true}'], temperature=0.0)
    assert provider.generate(MESSAGES, json_format=True).text == '{"ok": true}'
    assert calls(provider)[0]["response_format"] == {"type": "json_object"} and calls(provider)[0]["temperature"] == 0.0
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    provider.generate(MESSAGES, json_schema=schema)
    assert calls(provider)[1]["response_format"]["json_schema"]["schema"] == schema
    # an endpoint without response_format: JSON asked for in the prompt, code fences removed
    plain = openai_provider(['```json\n{"ok": true}\n```'], json_mode=False)
    assert plain.generate(MESSAGES, json_format=True).text == '{"ok": true}'
    sent = calls(plain)[0]
    assert "response_format" not in sent and JSON_ONLY in sent["messages"][0]["content"]


def test_openai_compatible_errors_become_llm_errors():
    with pytest.raises(LLMError, match="mistral request failed"):
        openai_provider([], error=openai.OpenAIError("boom")).generate(MESSAGES)


class FakeStream:
    """The context manager messages.stream() returns: text pieces, then a final message."""

    def __init__(self, pieces, final):
        """Pieces to stream and the final message object."""
        self.text_stream, self.final = iter(pieces), final

    def __enter__(self):
        """Enter the stream."""
        return self

    def __exit__(self, *exc):
        """Leave the stream."""
        return False

    def get_final_message(self):
        """The accumulated message (usage, stop reason)."""
        return self.final


def anthropic_provider(pieces, stop_reason="end_turn", **kwargs):
    """An AnthropicProvider whose client is a fake recording the stream() arguments."""
    provider = AnthropicProvider("key", "claude-opus-5-5", name="claude-opus", **kwargs)
    final = SimpleNamespace(stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=20, output_tokens=5),
                            stop_details=SimpleNamespace(category="cyber"))
    provider.sent = []

    def stream(**kwargs):
        """Record the request and stream the pieces."""
        provider.sent.append(kwargs)
        return FakeStream(pieces, final)

    provider.client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    return provider


def test_anthropic_moves_the_system_message_and_streams():
    seen = []
    provider = anthropic_provider(["Port ", "9040 [1]."], effort="medium")
    g = provider.generate(MESSAGES, on_token=seen.append)
    assert (g.text, g.provider, g.prompt_tokens, g.output_tokens) == ("Port 9040 [1].", "claude-opus", 20, 5)
    sent = provider.sent[0]
    assert sent["system"] == "Rules." and sent["messages"] == [{"role": "user", "content": "Question?"}]
    assert sent["output_config"] == {"effort": "medium"} and "temperature" not in sent
    assert sent["max_tokens"] == 4000 and seen == ["Port ", "9040 [1]."]


def test_anthropic_json_with_and_without_a_schema():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
              "additionalProperties": False}
    provider = anthropic_provider(['{"ok": true}'])
    provider.generate(MESSAGES, json_schema=schema)
    assert provider.sent[0]["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    fenced = anthropic_provider(['```json\n{"ok": true}\n```'])
    assert fenced.generate(MESSAGES, json_format=True).text == '{"ok": true}'
    assert JSON_ONLY in fenced.sent[0]["system"] and "output_config" not in fenced.sent[0]


def test_anthropic_refusal_is_an_llm_error():
    with pytest.raises(LLMError, match="declined the request \\(refusal: cyber\\)"):
        anthropic_provider([""], stop_reason="refusal").generate(MESSAGES)


def test_make_provider_builds_the_adapter_of_each_profile(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret")
    local = make_provider(ModelProfile("local", "ollama", "qwen2.5:7b-instruct", base_url="http://h:11434",
                                       context_tokens=4096, max_output_tokens=500))
    assert isinstance(local, OllamaProvider) and local.options == {"num_ctx": 4096, "temperature": 0.0, "num_predict": 500}
    gem = make_provider(ModelProfile("gemini", "openai_compatible", "gemini-3.8-flash", location="external",
                                     base_url="https://g/openai/", api_key_env="TEST_KEY", json_mode=False))
    assert isinstance(gem, OpenAIProvider) and (gem.name, gem.base_url, gem.json_mode) == ("gemini", "https://g/openai/", False)
    claude = make_provider(ModelProfile("claude", "anthropic", "claude-opus-5-5", location="external",
                                        api_key_env="TEST_KEY", max_output_tokens=8000, effort="high"))
    assert isinstance(claude, AnthropicProvider) and (claude.max_tokens, claude.effort) == (8000, "high")


def test_strip_json_fences():
    assert strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_json_fences('  {"a": 1} ') == '{"a": 1}'
