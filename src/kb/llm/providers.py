"""LLM providers (adapters): one class per way of talking to a model, all with the same interface.

    OllamaProvider     local models through the Ollama server
    OpenAIProvider     the OpenAI API and every OpenAI-compatible endpoint (Mistral, Gemini, local
                       servers such as LM Studio or vLLM), chosen by base_url
    AnthropicProvider  Claude through the official anthropic SDK

All stream: `generate` calls `on_token` for every piece of text as it arrives and returns the full
text with token counts and timings in a `Generation`; every failure becomes an `LLMError`.
`json_format=True` asks for a reply that is one JSON object (with `json_schema`, one that follows
the schema where the provider supports it). `make_provider()` builds the adapter for a catalogue
profile (kb.llm.catalogue). Which model may see which context is decided by the model registry
(kb.llm.registry): text goes to an external model only when every context source has external_ok.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import httpx

if TYPE_CHECKING:
    from kb.llm.catalogue import ModelProfile

OLLAMA = "ollama"
OPENAI = "openai"
ANTHROPIC = "anthropic"
JSON_ONLY = "Reply with one JSON object only: no code fences, no text before or after it."
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass
class Generation:
    """The LLM's full answer with provider, model, token counts and timings."""
    text: str
    provider: str
    model: str
    seconds: float                       # wall time of the call, including model loading
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    load_seconds: float | None = None    # Ollama: time spent loading the model into memory
    eval_seconds: float | None = None    # time spent generating the output tokens

    @property
    def tokens_per_s(self) -> float | None:
        """Generation speed in output tokens per second (None if unknown)."""
        if not self.output_tokens:
            return None
        seconds = self.eval_seconds or self.seconds
        return self.output_tokens / seconds if seconds else None


class LLMProvider(Protocol):
    """Interface of an LLM provider: a name, a model and a streaming `generate`."""
    name: str
    model: str

    def generate(self, messages: list[dict], *, on_token: Callable[[str], None] | None = None,
                 json_format: bool = False, json_schema: dict | None = None) -> Generation:
        """Generate a reply to the chat messages, calling `on_token` with each piece as it streams."""


class LLMError(RuntimeError):
    """The provider could not be reached or rejected the request; the message says what to do."""


def strip_json_fences(text: str) -> str:
    """The JSON inside a ```json … ``` block (models asked for JSON in the prompt sometimes add one)."""
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


def _with_json_instruction(messages: list[dict]) -> list[dict]:
    """The messages with JSON_ONLY added to the system message (or as one), for prompt-only JSON."""
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": f"{messages[0]['content']}\n\n{JSON_ONLY}"}, *messages[1:]]
    return [{"role": "system", "content": JSON_ONLY}, *messages]


class OllamaProvider:
    """Local LLM through the Ollama server (default provider)."""
    name = OLLAMA

    def __init__(self, host: str, model: str, *, num_ctx: int = 8192, keep_alive: str = "10m",
                 temperature: float = 0.0, max_tokens: int = 1500, timeout: float = 600, name: str = OLLAMA):
        """Client for the Ollama server at `host`, with context size, temperature and answer cap.
        name: the catalogue profile it serves (shown in traces)."""
        from ollama import Client

        self.name = name
        self.host = host
        self.model = model
        self.options = {"num_ctx": num_ctx, "temperature": temperature, "num_predict": max_tokens}
        self.keep_alive = keep_alive
        self.client = Client(host=host, timeout=timeout)

    def generate(self, messages: list[dict], *, on_token: Callable[[str], None] | None = None,
                 json_format: bool = False, json_schema: dict | None = None) -> Generation:
        """Stream a chat reply from Ollama; raises LLMError if it is unreachable or the model is missing.

        json_format: constrain the reply to valid JSON (comparison split, faithfulness judge);
        json_schema: constrain it to that schema instead.
        """
        from ollama import ResponseError

        start = time.perf_counter()
        parts, final = [], None
        try:
            for part in self.client.chat(self.model, messages, stream=True, options=self.options,
                                         keep_alive=self.keep_alive,
                                         format=json_schema or ("json" if json_format else None)):
                piece = part.message.content or ""
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
                if part.done:
                    final = part
        except ResponseError as e:
            hint = f"; run `ollama pull {self.model}`" if e.status_code == 404 else ""
            raise LLMError(f"Ollama rejected the request: {e.error}{hint}") from e
        except (ConnectionError, httpx.ConnectError) as e:
            raise LLMError(f"Ollama is not reachable at {self.host}; start Ollama and retry") from e
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama did not answer in time ({type(e).__name__})") from e

        def seconds(ns: int | None) -> float | None:
            """Ollama reports durations in nanoseconds; convert to seconds (None if absent)."""
            return ns / 1e9 if ns else None

        return Generation(
            text="".join(parts), provider=self.name, model=self.model, seconds=time.perf_counter() - start,
            prompt_tokens=final.prompt_eval_count if final else None,
            output_tokens=final.eval_count if final else None,
            load_seconds=seconds(final.load_duration) if final else None,
            eval_seconds=seconds(final.eval_duration) if final else None,
        )


class OpenAIProvider:
    """Chat completions on the OpenAI API or any OpenAI-compatible endpoint (base_url): Mistral,
    Gemini, LM Studio, vLLM. Temperature is sent only when set: several current models accept only
    their default value."""

    name = OPENAI

    def __init__(self, api_key: str, model: str, *, max_tokens: int = 1500, timeout: float = 120,
                 base_url: str = "", name: str = OPENAI, temperature: float | None = None, json_mode: bool = True):
        """Client for `model` at `base_url` (empty: the OpenAI API), with an answer cap, a timeout and
        one retry. json_mode: the endpoint accepts response_format (otherwise JSON is asked for in the prompt)."""
        from openai import OpenAI

        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None, timeout=timeout,
                             max_retries=1)

    def generate(self, messages: list[dict], *, on_token: Callable[[str], None] | None = None,
                 json_format: bool = False, json_schema: dict | None = None) -> Generation:
        """Stream a chat completion; raises LLMError on any API failure."""
        import openai

        start = time.perf_counter()
        parts, usage = [], None
        extra: dict = {}
        if self.temperature is not None:
            extra["temperature"] = self.temperature
        if (json_format or json_schema) and self.json_mode:
            extra["response_format"] = ({"type": "json_schema", "json_schema": {"name": "reply", "schema": json_schema}}
                                        if json_schema else {"type": "json_object"})
        elif json_format or json_schema:
            messages = _with_json_instruction(messages)
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True, max_completion_tokens=self.max_tokens,
                stream_options={"include_usage": True}, **extra,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    piece = chunk.choices[0].delta.content
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
                if chunk.usage:
                    usage = chunk.usage
        except openai.AuthenticationError as e:
            raise LLMError(f"{self.name}: the API key was rejected; check the key variable in .env") from e
        except openai.RateLimitError as e:
            raise LLMError(f"{self.name}: rate limit or quota reached; retry later") from e
        except openai.NotFoundError as e:
            raise LLMError(f"{self.name}: model {self.model!r} or endpoint not found ({e})") from e
        except openai.APIConnectionError as e:
            raise LLMError(f"{self.name}: not reachable at {self.base_url or 'the OpenAI API'} ({e})") from e
        except openai.OpenAIError as e:
            raise LLMError(f"{self.name} request failed: {e}") from e
        text = "".join(parts)
        return Generation(
            text=strip_json_fences(text) if (json_format or json_schema) else text, provider=self.name,
            model=self.model, seconds=time.perf_counter() - start,
            prompt_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


class AnthropicProvider:
    """Claude through the official anthropic SDK (Messages API, streamed).

    The system message goes in the `system` field. Current Claude models think adaptively and reject
    sampling settings, so no temperature is sent; `effort` (low … max) sets the depth instead.
    JSON: with a schema, structured output (`output_config.format`); without one, asked for in the
    prompt. A refusal (stop_reason "refusal") becomes an LLMError, so a fallback model can answer.
    """

    name = ANTHROPIC

    def __init__(self, api_key: str, model: str, *, max_tokens: int = 4000, timeout: float = 120,
                 effort: str = "", name: str = ANTHROPIC):
        """Client for `model`, with an output cap (thinking included), a timeout and the SDK's retries."""
        from anthropic import Anthropic

        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = Anthropic(api_key=api_key, timeout=timeout)

    def generate(self, messages: list[dict], *, on_token: Callable[[str], None] | None = None,
                 json_format: bool = False, json_schema: dict | None = None) -> Generation:
        """Stream a reply from Claude; raises LLMError on any API failure or a refusal."""
        import anthropic

        if json_format and not json_schema:
            messages = _with_json_instruction(messages)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        extra: dict = {"output_config": output_config} if output_config else {}
        if system:
            extra["system"] = system

        start = time.perf_counter()
        parts = []
        try:
            with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens, messages=chat,
                                             **extra) as stream:
                for piece in stream.text_stream:
                    if piece:
                        parts.append(piece)
                        if on_token:
                            on_token(piece)
                final = stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise LLMError(f"{self.name}: the API key was rejected; check the key variable in .env") from e
        except anthropic.RateLimitError as e:
            raise LLMError(f"{self.name}: rate limit reached; retry later") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"{self.name}: model {self.model!r} not found ({e})") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"{self.name} request failed ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"{self.name}: the Anthropic API is not reachable ({e})") from e
        if final.stop_reason == "refusal":
            category = getattr(getattr(final, "stop_details", None), "category", None)
            raise LLMError(f"{self.name} declined the request (refusal{f': {category}' if category else ''})")
        text = "".join(parts)
        return Generation(
            text=strip_json_fences(text) if json_format or json_schema else text, provider=self.name,
            model=self.model, seconds=time.perf_counter() - start,
            prompt_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens,
        )


def make_provider(profile: "ModelProfile") -> LLMProvider:
    """The adapter for a catalogue profile (kb.llm.catalogue.ModelProfile)."""
    if profile.adapter == "ollama":
        return OllamaProvider(profile.base_url, profile.model, num_ctx=profile.context_tokens,
                              keep_alive=profile.keep_alive,
                              temperature=0.0 if profile.temperature is None else profile.temperature,
                              max_tokens=profile.max_output_tokens, timeout=profile.timeout, name=profile.name)
    if profile.adapter == "openai_compatible":
        return OpenAIProvider(profile.api_key, profile.model, max_tokens=profile.max_output_tokens,
                              timeout=profile.timeout, base_url=profile.base_url, name=profile.name,
                              temperature=profile.temperature, json_mode=profile.json_mode)
    if profile.adapter == "anthropic":
        return AnthropicProvider(profile.api_key, profile.model, max_tokens=profile.max_output_tokens,
                                 timeout=profile.timeout, effort=profile.effort, name=profile.name)
    raise ValueError(f"unknown adapter {profile.adapter!r}")
