"""bge-m3 embeddings (dense + learned sparse), shared by ingestion (GPU) and queries (CPU)."""

from dataclasses import dataclass
from typing import Protocol

from kb.core.config import get_settings

# Identifies the embedding setup; part of each document's index_version.
EMBEDDING_ID = "bge-m3-ds1"


@dataclass
class Embedding:
    """One text's bge-m3 vectors: dense (1024 floats) and sparse (token ids with weights)."""
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder(Protocol):
    """Anything that turns texts into embeddings (bge-m3 in production, a fake in tests)."""

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed the texts, returning one Embedding per text in the same order."""
        ...


class BgeM3Embedder:
    """bge-m3 via FlagEmbedding. GPU (fp16) when available unless a device is given."""

    def __init__(self, device: str | None = None, batch_size: int = 16, max_length: int = 1024):
        """Load bge-m3 on `device` (GPU if available, else CPU; fp16 on GPU) and warm it up."""
        import torch
        from FlagEmbedding import BGEM3FlagModel

        from kb.core.progress import silence_progress_bars

        silence_progress_bars()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = BGEM3FlagModel(str(get_settings().embed_model_path),
                                    use_fp16=self.device != "cpu", devices=self.device)
        # The first encode initialises CUDA kernels (~30 s on GPU); do it at load time, not in the first batch.
        self.model.encode(["warm-up"], return_dense=True, return_sparse=True, return_colbert_vecs=False)

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Dense + sparse embeddings for the texts, encoded in batches; zero-weight tokens dropped."""
        if not texts:
            return []
        out = self.model.encode(texts, batch_size=self.batch_size, max_length=self.max_length,
                                return_dense=True, return_sparse=True, return_colbert_vecs=False)
        result = []
        for dense, weights in zip(out["dense_vecs"], out["lexical_weights"], strict=True):
            items = sorted((int(token), float(weight)) for token, weight in weights.items() if weight > 0)
            result.append(Embedding([float(x) for x in dense], [t for t, _ in items], [w for _, w in items]))
        return result
