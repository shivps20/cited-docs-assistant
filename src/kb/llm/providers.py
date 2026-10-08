"""LLM providers: Ollama (local, default) and OpenAI (optional, external).

Both stream: `generate` calls `on_token` for every piece of text as it arrives and returns the
full text with token counts and timings. Which provider may see which context is decided in
`select_provider`: text goes to OpenAI only when every context source has external_ok.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from kb.core.config import Settings
from kb.retrieve.assemble import ContextUnit

OLLAMA = "ollama"
OPENAI = "openai"


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

    def generate(self, messages: list[dict], *, on_token: Callable[[str], None] | None = None) -> Generation:
        """Generate a reply to the chat messages, calling `on_token` with each piece as it streams."""


class LLMError(RuntimeError):
    """The provider could not be reached or rejected the request; the message says what to do."""


class OllamaProvider:
    """Local LLM through the Ollama server (default provider)."""
    name = OLLAMA

    def __init__(self, host: str, model: str, *, num_ctx: int = 8192, keep_alive: str = "10m",
                 temperature: float = 0.0, max_tokens: int = 1500, timeout: float = 600):
        """Client for the Ollama server at `host`, with context size, temperature and answer cap."""
        from ollama import Client

        self.host = host
        self.model = model
        self.options = {"num_ctx": num_ctx, "temperature": temperature, "num_predict": max_tokens}
        self.keep_alive = keep_alive
        self.client = Client(host=host, timeout=timeout)

    def generate(self, messages: list[dict], *, on_token: Callable[[str], None] | None = None,
                 json_format: bool = False) -> Generation:
        """Stream a chat reply from Ollama; raises LLMError if it is unreachable or the model is missing.

        json_format: constrain the reply to valid JSON (used by the faithfulness judge).
        """
        from ollama import ResponseError

        start = time.perf_counter()
        parts, final = [], None
        try:
            for part in self.client.chat(self.model, messages, stream=True, options=self.options,
                                         keep_alive=self.keep_alive, format="json" if json_format else None):
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
    """OpenAI chat completions. Temperature is left at the model's default: several current
    models accept only their default value."""

    name = OPENAI

    def __init__(self, api_key: str, model: str, *, max_tokens: int = 1500, timeout: float = 120):
        """OpenAI client for `model`, with an answer cap, a timeout and one retry."""
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=1)

    def generate(self, messages: list[dict], *, on_token: Callable[[str], None] | None = None) -> Generation:
        """Stream a chat completion from OpenAI; raises LLMError on any API failure."""
        import openai

        start = time.perf_counter()
        parts, usage = [], None
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True, max_completion_tokens=self.max_tokens,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    piece = chunk.choices[0].delta.content
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
                if chunk.usage:
                    usage = chunk.usage
        except openai.OpenAIError as e:
            raise LLMError(f"OpenAI request failed: {e}") from e
        return Generation(
            text="".join(parts), provider=self.name, model=self.model, seconds=time.perf_counter() - start,
            prompt_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


def build_providers(settings: Settings) -> dict[str, LLMProvider]:
    """Ollama always; OpenAI when both OPENAI_API_KEY and OPENAI_MODEL are set."""
    providers: dict[str, LLMProvider] = {
        OLLAMA: OllamaProvider(settings.ollama_host, settings.llm_model, num_ctx=settings.llm_num_ctx,
                               keep_alive=settings.llm_keep_alive, temperature=settings.llm_temperature,
                               max_tokens=settings.llm_max_tokens),
    }
    if settings.openai_api_key and settings.openai_model:
        providers[OPENAI] = OpenAIProvider(settings.openai_api_key, settings.openai_model,
                                           max_tokens=settings.llm_max_tokens)
    return providers


def select_provider(requested: str, context: Sequence[ContextUnit], available: Sequence[str]
                    ) -> tuple[str, str | None]:
    """(provider name, notice for the user or None).

    auto    OpenAI when configured and every context source has external_ok, else Ollama.
    ollama  always Ollama.
    openai  OpenAI if every source has external_ok; otherwise Ollama, with a notice.
    """
    if requested not in ("auto", OLLAMA, OPENAI):
        raise ValueError(f"unknown provider {requested!r}")
    if requested == OLLAMA:
        return OLLAMA, None
    if OPENAI not in available:
        if requested == OPENAI:
            raise LLMError("OpenAI is not configured: set OPENAI_API_KEY and OPENAI_MODEL in .env")
        return OLLAMA, None
    blocked = sorted({u.doc_id for u in context if not u.external_ok})
    if blocked:
        return OLLAMA, (f"Answered with the local model: {', '.join(blocked)} may not be sent to an "
                        "external LLM (external_ok = false).")
    return OPENAI, None
