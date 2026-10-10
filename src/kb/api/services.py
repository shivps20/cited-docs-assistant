"""Everything the API loads once at startup and shares between requests.

Models (bge-m3, the reranker) and the LLM providers are expensive to load, so they live here for
the life of the server. SQLite connections are not: every request opens its own (`connect()`),
because a connection cannot be shared across the threads FastAPI runs requests on.

`model_lock` serialises the work that uses the models: the CPU models are not built for parallel
calls, and the 6 GB GPU holds one LLM generation at a time.
"""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from kb.answer.pipeline import Answerer
from kb.api.users import UserDirectory
from kb.core.config import Settings
from kb.core.db import connect
from kb.llm.catalogue import CatalogueError, load_catalogue
from kb.llm.providers import LLMProvider
from kb.llm.registry import ModelRegistry
from kb.retrieve.pipeline import Retriever
from kb.retrieve.rerank import Reranker
from kb.store.embed import Embedder

log = logging.getLogger(__name__)
CONDENSE_MAX_TOKENS = 96     # a rewritten question is one line


@dataclass
class Services:
    """Shared, long-lived objects for all requests."""

    settings: Settings
    users: UserDirectory
    client: QdrantClient | Any
    embedder: Embedder | None
    reranker: Reranker | None
    models: ModelRegistry                     # catalogue models, roles, privacy policy (kb.llm.registry)
    condenser: LLMProvider | None = None      # the condenser role's model, with a short answer cap
    load_seconds: float = 0.0
    model_lock: threading.Lock = field(default_factory=threading.Lock)
    models_stamp: float | None = None         # models.yaml modification time when it was loaded
    reload_lock: threading.Lock = field(default_factory=threading.Lock)

    def current_models(self) -> ModelRegistry:
        """The model registry, re-read when models.yaml changed since it was loaded (no restart needed).

        A catalogue with errors is not taken over: the previous one stays, and the problem is logged.
        """
        path = self.settings.models_path
        stamp = path.stat().st_mtime if path.exists() else None
        if stamp == self.models_stamp:
            return self.models
        with self.reload_lock:
            if stamp != self.models_stamp:
                try:
                    models = ModelRegistry(load_catalogue(path, self.settings))
                    self.models = models
                    self.condenser = models.for_role("condenser", max_output_tokens=CONDENSE_MAX_TOKENS)
                    log.info("models.yaml reloaded: answer model %s", models.catalogue.roles["answer"])
                except CatalogueError as e:
                    log.warning("models.yaml changed but is invalid; keeping the previous catalogue:\n%s", e)
                self.models_stamp = stamp
        return self.models

    def connect(self) -> sqlite3.Connection:
        """A new SQLite connection for one request (schema checked; caller closes it)."""
        return connect(self.settings.db_path)

    def make_answerer(self, conn: sqlite3.Connection) -> Answerer:
        """An Answerer for one request: shared models and providers, this request's connection."""
        retriever = Retriever(conn, self.client, self.settings.qdrant_collection, self.embedder, self.reranker)
        return Answerer(conn, retriever, self.current_models(), not_found_score=self.settings.not_found_score,
                        provider=self.settings.llm_provider, refusal_retry=self.settings.refusal_retry,
                        cache=self.settings.answer_cache)


def load_services(settings: Settings) -> Services:
    """Load users, connect to Qdrant and load bge-m3, the reranker and the LLM providers (CPU models)."""
    from kb.api.users import load_users
    from kb.retrieve.rerank import BgeReranker
    from kb.store.embed import BgeM3Embedder
    from kb.store.vectorstore import get_client

    users = load_users(settings.users_path)          # fail fast on a bad users.yaml
    start = time.perf_counter()
    embedder = BgeM3Embedder(device="cpu")
    reranker = BgeReranker(device="cpu")
    models = ModelRegistry(load_catalogue(settings.models_path, settings))   # fail fast on a bad models.yaml
    stamp = settings.models_path.stat().st_mtime if settings.models_path.exists() else None
    # The condenser role's model with a short answer cap; for a local model keep the same num_ctx as
    # answering (Ollama reloads the model when num_ctx changes).
    condenser = models.for_role("condenser", max_output_tokens=CONDENSE_MAX_TOKENS)
    return Services(settings=settings, users=users, client=get_client(), embedder=embedder, reranker=reranker,
                    models=models, condenser=condenser, models_stamp=stamp,
                    load_seconds=time.perf_counter() - start)
