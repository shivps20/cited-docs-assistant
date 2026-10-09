"""Health checks behind GET /api/health: SQLite, Qdrant, Ollama and the loaded models.

Each check returns {"ok": bool, "detail": str}; the overall status is "ok" only if all pass.
Checks are quick (short timeouts) so the endpoint can be polled by the UI.
"""

import httpx

from kb.api.services import Services
from kb.core.db import SchemaError


def sqlite_status(services: Services) -> dict:
    """The database opens and its schema is current."""
    try:
        conn = services.connect()
    except SchemaError as e:
        return {"ok": False, "detail": str(e)}
    try:
        docs = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'indexed'").fetchone()[0]
    finally:
        conn.close()
    return {"ok": True, "detail": f"{docs} indexed documents"}


def qdrant_status(services: Services) -> dict:
    """The collection exists; reports its point count."""
    name = services.settings.qdrant_collection
    try:
        if not services.client.collection_exists(name):
            return {"ok": False, "detail": f"collection '{name}' missing; run scripts/init_qdrant.py"}
        points = services.client.count(name, exact=True).count
    except Exception as e:  # noqa: BLE001 - any client/connection error means "not healthy"
        return {"ok": False, "detail": f"{services.settings.qdrant_url} unreachable ({type(e).__name__})"}
    return {"ok": True, "detail": f"'{name}' holds {points} points"}


def ollama_status(host: str, model: str) -> dict:
    """Ollama answers and has the configured model pulled."""
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=2).json().get("models", [])
    except httpx.HTTPError as e:
        return {"ok": False, "detail": f"{host} unreachable ({type(e).__name__}); start Ollama"}
    if model not in {m.get("name") for m in tags}:
        return {"ok": False, "detail": f"model '{model}' not pulled; run `ollama pull {model}`"}
    return {"ok": True, "detail": f"'{model}' available"}


def models_status(services: Services) -> dict:
    """The CPU models are loaded; names the answer model and the models whose key is set."""
    loaded = services.embedder is not None and services.reranker is not None
    catalogue = services.models.catalogue
    ready = [name for name in catalogue.models if services.models.ready(name)]
    detail = (f"bge-m3 + reranker loaded in {services.load_seconds:.1f} s; answer model: {catalogue.roles['answer']}; "
              f"models ready: {', '.join(ready) or 'none'}") if loaded else "embedding / reranking models not loaded"
    return {"ok": loaded and bool(ready), "detail": detail}


def check_health(services: Services) -> dict:
    """All checks plus the overall status ('ok' or 'degraded')."""
    s = services.settings
    checks = {
        "sqlite": sqlite_status(services),
        "qdrant": qdrant_status(services),
        "ollama": ollama_status(s.ollama_host, s.llm_model),
        "models": models_status(services),
    }
    return {"status": "ok" if all(c["ok"] for c in checks.values()) else "degraded", "checks": checks}
