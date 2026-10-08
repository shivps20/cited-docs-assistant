"""Health check for everything the knowledge-base assistant depends on.

    uv run python scripts/check_env.py           # standard checks (~30 s, mostly importing torch)
    uv run python scripts/check_env.py --deep    # also loads the LLM and runs a 1-token generation

Exit code 1 if any check FAILs; WARNs don't fail.
"""

import argparse
import os
import shutil
import sqlite3
import sys

import httpx

from kb.core.config import ROOT, get_settings
from kb.core.db import SCHEMA_VERSION
from kb.ingest.manifest import ManifestError, find_unlisted, load_manifest
from kb.store import vectorstore  # expected collection config

ENV_FILE = ROOT / ".env"
SETTINGS = get_settings()
QDRANT_URL = SETTINGS.qdrant_url
OLLAMA_HOST = SETTINGS.ollama_host
LLM_MODEL = SETTINGS.llm_model
LLM_NUM_CTX = SETTINGS.llm_num_ctx

# Files each model folder must contain for FlagEmbedding / Docling to load it.
MODEL_FILES = {
    SETTINGS.embed_model_path: ["pytorch_model.bin", "sparse_linear.pt", "colbert_linear.pt", "tokenizer.json"],
    SETTINGS.rerank_model_path: ["model.safetensors", "tokenizer.json"],
    SETTINGS.docling_artifacts_path: ["docling-project--docling-models", "docling-project--docling-layout-heron"],
}
MIN_FREE_DISK_GB = 20

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str) -> None:
    """Add one check result (PASS / WARN / FAIL / INFO) to the report."""
    results.append((status, name, detail))


def check_env_file() -> None:
    """Is there a .env file? Without one, the built-in defaults are used."""
    if ENV_FILE.exists():
        record("PASS", ".env", str(ENV_FILE.name))
    else:
        record("WARN", ".env", "missing; using defaults (copy .env.example to .env)")


def check_qdrant() -> None:
    """Qdrant reachable, client/server versions compatible, collection vectors and payload indexes correct.
    """
    try:
        httpx.get(f"{QDRANT_URL}/healthz", timeout=3).raise_for_status()
        server = httpx.get(QDRANT_URL, timeout=3).json()["version"]
    except httpx.HTTPError as e:
        record("FAIL", "qdrant", f"{QDRANT_URL} unreachable ({type(e).__name__}); run `docker compose -f deploy/docker-compose.yml up -d`")
        return

    from importlib.metadata import version

    from qdrant_client import QdrantClient

    client_ver = version("qdrant-client")
    same_minor = server.split(".")[:2] == client_ver.split(".")[:2]
    record("PASS" if same_minor else "WARN", "qdrant server",
           f"v{server} at {QDRANT_URL}, client v{client_ver}" + ("" if same_minor else " (minor version mismatch)"))

    client = QdrantClient(url=QDRANT_URL)
    name = SETTINGS.qdrant_collection
    if not client.collection_exists(name):
        record("FAIL", "qdrant collection", f"'{name}' missing; run scripts/init_qdrant.py")
        return
    info = client.get_collection(name)
    params = info.config.params
    dense = params.vectors.get(vectorstore.DENSE) if isinstance(params.vectors, dict) else None
    sparse = (params.sparse_vectors or {}).get(vectorstore.SPARSE)
    missing_idx = sorted(set(vectorstore.PAYLOAD_INDEXES) - set(info.payload_schema))
    problems = []
    if dense is None or dense.size != vectorstore.DENSE_DIM:
        problems.append(f"dense vector not {vectorstore.DENSE_DIM}-dim")
    if sparse is None:
        problems.append("sparse vector missing")
    if missing_idx:
        problems.append(f"missing indexes {missing_idx}")
    if problems:
        record("FAIL", "qdrant collection", f"'{name}': {'; '.join(problems)}; run scripts/init_qdrant.py")
    else:
        record("PASS", "qdrant collection",
               f"'{name}' status={info.status.value} points={info.points_count} indexes={len(info.payload_schema)}")


def check_sqlite() -> None:
    """Database file present, schema at the current version, WAL mode and integrity check passing."""
    path = SETTINGS.db_path
    if not path.exists():
        record("FAIL", "sqlite", f"{path} missing; run scripts/init_db.py")
        return
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        ok = conn.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        conn.close()
    problems = []
    if ver != SCHEMA_VERSION:
        problems.append(f"schema v{ver}, expected v{SCHEMA_VERSION} (run scripts/init_db.py)")
    if mode != "wal":
        problems.append(f"journal_mode={mode}, expected wal")
    if ok != "ok":
        problems.append(f"quick_check: {ok}")
    if problems:
        record("FAIL", "sqlite", "; ".join(problems))
    else:
        record("PASS", "sqlite", f"{path.relative_to(ROOT)} schema v{ver}, wal, integrity ok")


def check_ollama(deep: bool) -> None:
    """Ollama reachable and the LLM pulled; with `deep`, also load the model and generate a reply."""
    try:
        ver = httpx.get(f"{OLLAMA_HOST}/api/version", timeout=3).json()["version"]
        tags = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json()["models"]
    except httpx.HTTPError as e:
        record("FAIL", "ollama", f"{OLLAMA_HOST} unreachable ({type(e).__name__}); start the Ollama app")
        return
    names = {m["name"] for m in tags}
    if LLM_MODEL not in names:
        record("FAIL", "ollama", f"v{ver}, model '{LLM_MODEL}' not pulled; run `ollama pull {LLM_MODEL}`")
        return
    record("PASS", "ollama", f"v{ver}, '{LLM_MODEL}' available")

    if deep:
        try:
            r = httpx.post(f"{OLLAMA_HOST}/api/generate", timeout=180, json={
                "model": LLM_MODEL, "prompt": "Reply with OK.", "stream": False,
                "options": {"num_ctx": LLM_NUM_CTX, "num_predict": 3}, "keep_alive": 0,
            })
            r.raise_for_status()
            body = r.json()
            load_s = body.get("load_duration", 0) / 1e9
            record("PASS", "ollama generate", f"num_ctx={LLM_NUM_CTX}, load {load_s:.1f}s, reply={body['response'].strip()!r}")
        except httpx.HTTPError as e:
            record("FAIL", "ollama generate", f"{type(e).__name__}: {e}")


def check_gpu() -> None:
    """CUDA build of torch, GPU visible, and enough free VRAM reported."""
    import torch

    if not torch.cuda.is_available():
        build = torch.__version__
        record("FAIL", "gpu", f"torch {build}: CUDA not available" + (" (CPU-only build)" if "+cpu" in build or "+cu" not in build else ""))
        return
    free, total = torch.cuda.mem_get_info()
    status = "PASS" if free / total > 0.5 else "WARN"
    note = "" if status == "PASS" else " (another process holds VRAM, e.g. a loaded Ollama model)"
    record(status, "gpu", f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
                          f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free{note}")


def check_models() -> None:
    """Every local model directory is complete, and HF_HUB_OFFLINE keeps libraries offline."""
    for path, required in MODEL_FILES.items():
        missing = [f for f in required if not (path / f).exists()]
        if missing:
            record("FAIL", f"model {path.name}", f"missing {missing} in {path}; run scripts/download_models.py")
        else:
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30
            record("PASS", f"model {path.name}", f"{size:.1f} GiB at {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    if os.getenv("HF_HUB_OFFLINE") != "1":
        record("WARN", "hf offline", "HF_HUB_OFFLINE is not 1; libraries may try to reach Hugging Face at runtime")


def check_manifest() -> None:
    """The manifest loads without errors; warns about files in the documents folder it does not list."""
    try:
        docs = load_manifest(SETTINGS.manifest_path, SETTINGS.docs_dir)
    except FileNotFoundError:
        record("FAIL", "manifest", f"{SETTINGS.manifest_path} missing")
        return
    except ManifestError as e:
        record("FAIL", "manifest", f"{len(e.errors)} error(s); run `uv run kb manifest validate`")
        return
    unlisted = find_unlisted(SETTINGS.manifest_path, SETTINGS.docs_dir)
    status = "WARN" if unlisted else "PASS"
    record(status, "manifest", f"{len(docs)} documents" + (f", {len(unlisted)} files not in manifest "
                                                           "(run `uv run kb manifest scan`)" if unlisted else ""))


def check_openai() -> None:
    """Report whether the optional OpenAI provider is configured (key and model)."""
    key, model = SETTINGS.openai_api_key, SETTINGS.openai_model
    if not key:
        record("INFO", "openai", "not configured (local-only mode)")
    elif not model:
        record("WARN", "openai", "OPENAI_API_KEY set but OPENAI_MODEL empty")
    else:
        record("PASS", "openai", f"configured, model={model}")


def check_disk() -> None:
    """Enough free disk space for models and the parse cache."""
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    status = "PASS" if free_gb >= MIN_FREE_DISK_GB else "WARN"
    record(status, "disk", f"{free_gb:.0f} GB free on {ROOT.drive or ROOT.anchor}"
                           + ("" if status == "PASS" else f" (< {MIN_FREE_DISK_GB} GB; ingestion caches parsed docs)"))


def main() -> int:
    """Run all checks, print a PASS/WARN/FAIL table and return 1 if any check failed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deep", action="store_true", help="also load the LLM in Ollama and generate one token")
    args = parser.parse_args()

    checks = [check_env_file, check_qdrant, check_sqlite, lambda: check_ollama(args.deep),
              check_gpu, check_models, check_manifest, check_openai, check_disk]
    for check in checks:
        try:
            check()
        except Exception as e:  # noqa: BLE001 - a crashing check is a FAIL, not a crashed report
            record("FAIL", getattr(check, "__name__", "check"), f"{type(e).__name__}: {e}")

    width = max(len(name) for _, name, _ in results)
    for status, name, detail in results:
        print(f"[{status:<4}] {name:<{width}}  {detail}")

    fails = sum(s == "FAIL" for s, _, _ in results)
    warns = sum(s == "WARN" for s, _, _ in results)
    print(f"\n{'FAILED' if fails else 'OK'}: {fails} fail, {warns} warn")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
