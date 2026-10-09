"""Application settings, loaded from environment variables and the project .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]

# Also export .env into os.environ: third-party libraries read their own variables
# directly (e.g. HF_HUB_OFFLINE for huggingface_hub), which Settings alone would not set.
load_dotenv(ROOT / ".env", override=False)


class Settings(BaseSettings):
    """All configuration, read from environment variables and .env (environment wins), with defaults."""
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://127.0.0.1:6444"
    qdrant_collection: str = "kb_chunks"

    # Local storage; relative paths resolve from the project root
    db_path: Path = Field(Path("data/kb.core.db"), validation_alias="KB_DB_PATH")
    docs_dir: Path = Field(Path("data/documents"), validation_alias="KB_DOCS_DIR")
    manifest_path: Path = Field(Path("config/manifest.csv"), validation_alias="KB_MANIFEST_PATH")
    parsed_dir: Path = Field(Path("data/parsed"), validation_alias="KB_PARSED_DIR")
    users_path: Path = Field(Path("config/users.yaml"), validation_alias="KB_USERS_PATH")
    # Organisation-specific text rules and the golden question set (both local, git-ignored)
    domain_path: Path = Field(Path("config/domain.yaml"), validation_alias="KB_DOMAIN_PATH")
    golden_path: Path = Field(Path("eval/golden.json"), validation_alias="KB_GOLDEN_PATH")
    # Language models and which job each one does (Phase 6; local, git-ignored). Without the file:
    # the local Ollama model, plus OpenAI when OPENAI_* is set (kb.llm.catalogue.default_catalogue).
    models_path: Path = Field(Path("config/models.yaml"), validation_alias="KB_MODELS_PATH")

    # Chat API (kb serve). Loopback only by default: users are not authenticated.
    api_host: str = Field("127.0.0.1", validation_alias="KB_API_HOST")
    api_port: int = Field(8000, ge=1, le=65535, validation_alias="KB_API_PORT")

    # Local models
    embed_model_path: Path = Path("models/bge-m3")
    rerank_model_path: Path = Path("models/bge-reranker-v2-m3")
    docling_artifacts_path: Path = Path("models/docling")

    # Retrieval: rerank only the first N search candidates (CPU reranking cost grows with N)
    rerank_top: int = Field(20, ge=1, validation_alias="KB_RERANK_TOP")
    # Answer "not found" without calling the LLM when the top rerank score is below this.
    # Low on purpose: broad questions ('what are the different ... ?') can score 0.25 with the right
    # section at rank 1, and a wrong refusal loses the answer, while a wrong pass only costs an LLM
    # call that then refuses.
    not_found_score: float = Field(0.1, ge=0, le=1, validation_alias="KB_NOT_FOUND_SCORE")
    # When the LLM says "not found" although the gate passed, ask once more with only the
    # best-matching sources (TD-23: the 7B model gives up on long or mixed context; measured: 2 of 5
    # wrong refusals rescued, all unanswerable questions still refused).
    refusal_retry: bool = Field(True, validation_alias="KB_REFUSAL_RETRY")

    # LLM
    ollama_host: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_num_ctx: int = 8192
    llm_keep_alive: str = "10m"
    llm_temperature: float = 0.0             # 0: same answer for the same question and context
    llm_max_tokens: int = 1500               # cap on answer length (also bounds generation time)
    # auto: OpenAI when configured and every context source has external_ok, else Ollama
    llm_provider: Literal["auto", "ollama", "openai"] = Field("auto", validation_alias="KB_LLM_PROVIDER")
    openai_api_key: str = ""
    openai_model: str = ""

    @field_validator("db_path", "docs_dir", "manifest_path", "parsed_dir", "users_path", "domain_path",
                     "golden_path", "models_path", "embed_model_path",
                     "rerank_model_path", "docling_artifacts_path")
    @classmethod
    def _resolve_from_root(cls, path: Path) -> Path:
        """Resolve a relative path setting against the project root."""
        return path if path.is_absolute() else ROOT / path


@lru_cache
def get_settings() -> Settings:
    """The settings, loaded once per process (cached)."""
    return Settings()
