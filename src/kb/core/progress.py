"""Silence FlagEmbedding's per-call progress bars ('pre tokenize', 'Compute Scores', 'Inference Embeddings').

FlagEmbedding passes `disable=` to tqdm explicitly (shown whenever a call needs more than one
batch), so the TQDM_DISABLE environment variable has no effect. Instead, the `tqdm` / `trange`
names inside its modules are replaced by versions that are always disabled. Our own commands
report progress and timings themselves.
"""

import importlib

from tqdm import tqdm

_MODULES = (
    "FlagEmbedding.abc.inference.AbsEmbedder",
    "FlagEmbedding.abc.inference.AbsReranker",
    "FlagEmbedding.inference.embedder.encoder_only.base",
    "FlagEmbedding.inference.embedder.encoder_only.m3",
    "FlagEmbedding.inference.reranker.encoder_only.base",
)


def _quiet_tqdm(*args, **kwargs) -> tqdm:
    """tqdm that never draws, whatever `disable` the caller passed."""
    kwargs["disable"] = True
    return tqdm(*args, **kwargs)


def _quiet_trange(*args, **kwargs) -> tqdm:
    """trange (tqdm over range(*args)) that never draws."""
    kwargs["disable"] = True
    return tqdm(range(*args), **kwargs)


def silence_progress_bars() -> None:
    """Patch the FlagEmbedding modules that draw progress bars; safe to call more than once."""
    for name in _MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "tqdm"):
            module.tqdm = _quiet_tqdm
        if hasattr(module, "trange"):
            module.trange = _quiet_trange
