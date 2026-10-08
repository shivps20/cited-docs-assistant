"""Download the embedding, reranker and Docling models into ./models, then
smoke-test each one fully offline.

    uv run python scripts/download_models.py               # download + verify
    uv run python scripts/download_models.py --verify-only # verify only

Layout:
    models/bge-m3/              BAAI/bge-m3 (PyTorch weights only, no ONNX copy)
    models/bge-reranker-v2-m3/  BAAI/bge-reranker-v2-m3
    models/docling/             Docling layout / table / OCR models
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / os.getenv("KB_MODELS_DIR", "models")

HF_MODELS = {
    "BAAI/bge-m3": MODELS_DIR / "bge-m3",
    "BAAI/bge-reranker-v2-m3": MODELS_DIR / "bge-reranker-v2-m3",
}
# bge-m3 ships a 2.3 GB ONNX duplicate; FlagEmbedding only needs the PyTorch weights.
HF_IGNORE = ["onnx/*", "imgs/*", "*.onnx", "*.onnx_data"]
DOCLING_DIR = MODELS_DIR / "docling"
SAMPLE_PDF_DIR = ROOT / "data" / "documents"


def download() -> None:
    """Download each Hugging Face model and the Docling artifacts into ./models (network allowed)."""
    # .env sets HF_HUB_OFFLINE=1 for the app; downloading needs the network.
    # Must be set before huggingface_hub is first imported (it reads env at import).
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    for repo, target in HF_MODELS.items():
        print(f"\n== {repo} -> {target}")
        snapshot_download(repo_id=repo, local_dir=target, ignore_patterns=HF_IGNORE)

    print(f"\n== docling models -> {DOCLING_DIR}")
    cli = shutil.which("docling-tools", path=str(Path(sys.executable).parent))
    if cli is None:
        raise SystemExit("docling-tools not found in the venv; run `uv sync`")
    subprocess.run([cli, "models", "download", "-o", str(DOCLING_DIR)], check=True)


def verify_bge_m3() -> None:
    """Load bge-m3 offline and check it returns 1024-dim dense and non-empty sparse vectors."""
    import torch
    from FlagEmbedding import BGEM3FlagModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(str(HF_MODELS["BAAI/bge-m3"]), use_fp16=device != "cpu", devices=device)
    out = model.encode(
        ["What is the default session timeout?", "Configure the Oracle listener port."],
        return_dense=True, return_sparse=True,
    )
    dim = out["dense_vecs"].shape[1]
    assert dim == 1024, f"dense dim {dim} != 1024"
    top = sorted(out["lexical_weights"][0].items(), key=lambda kv: -kv[1])[:3]
    tokens = [model.tokenizer.decode([int(t)]) for t, _ in top]
    print(f"  bge-m3 ok on {device}: dense dim={dim}, top sparse tokens={tokens}")
    del model
    torch.cuda.empty_cache()


def verify_reranker() -> None:
    """Load the reranker offline and check it scores a relevant passage above an irrelevant one."""
    from FlagEmbedding import FlagReranker

    reranker = FlagReranker(str(HF_MODELS["BAAI/bge-reranker-v2-m3"]), use_fp16=False, devices="cpu")
    query = "What is the default session timeout?"
    good, bad = reranker.compute_score([
        [query, "The session timeout defaults to 30 minutes and can be changed in the server settings."],
        [query, "Oracle Database must be installed before running the application server installer."],
    ])
    assert good > bad, f"reranker ranked the irrelevant passage higher ({good:.2f} <= {bad:.2f})"
    print(f"  reranker ok on cpu: relevant={good:.2f} > irrelevant={bad:.2f}")


def verify_docling() -> None:
    """Convert a sample PDF with the local Docling models to prove they load offline."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf = next(SAMPLE_PDF_DIR.glob("*.pdf"), None)
    if pdf is None:
        print(f"  docling: skipped, no PDF in {SAMPLE_PDF_DIR}")
        return
    options = PdfPipelineOptions(artifacts_path=str(DOCLING_DIR), do_ocr=False, do_table_structure=True)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    start = time.perf_counter()
    doc = converter.convert(pdf, page_range=(1, 2)).document
    secs = time.perf_counter() - start
    headings = [t.text for t in doc.texts if t.label in ("section_header", "title")][:3]
    print(f"  docling ok: {pdf.name} pages 1-2 in {secs:.1f}s, {len(doc.texts)} text items, headings={headings}")


def verify() -> None:
    """Run all smoke tests with downloads forbidden, so a missing file fails loudly."""
    # Force offline so a missing file fails loudly instead of silently downloading.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name, check in [("bge-m3", verify_bge_m3), ("reranker", verify_reranker), ("docling", verify_docling)]:
        print(f"\n== verify {name}")
        check()


def main() -> int:
    """Download (unless --verify-only) and then verify every model; returns 1 on failure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true", help="skip downloads, only run the smoke tests")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        download()
    verify()
    print("\nall models ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
