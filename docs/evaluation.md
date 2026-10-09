# Evaluation

How retrieval and answers are measured against the golden question set (`eval/golden.json`, local; fictional example in [eval/golden.example.json](../eval/golden.example.json)). Ingestion coverage is checked with `kb coverage` (see [commands.md](commands.md)).

## `kb eval` — retrieval evaluation

| Option | Meaning |
|---|---|
| `--configs LIST` | Comma-separated subset of `dense`, `sparse`, `hybrid`, `hybrid+rr10`, `hybrid+rr15@256`, `hybrid+rr30` (default: all; the reranked ones take minutes) |
| `--questions LIST` | Only these golden IDs, e.g. `q001,q012` |
| `--misses` | List questions without a hit in the top 5 |

Prints Recall@1/5/10, MRR, context recall, all-docs@5 for comparisons, latency, scores for answerable vs unanswerable questions, and the access and release filter checks. The full report goes to `data/eval/retrieval-<timestamp>.json`.

```bash
uv run kb eval --configs dense,hybrid                 # fast (~10 s)
uv run kb eval --configs dense,hybrid+rr10 --misses   # with the reranker (~7 min)
uv run kb eval --questions q022,q040 --configs hybrid+rr10
uv run kb eval                                        # all six configurations (~20 min)
```

### Results behind the retrieval defaults

`uv run kb eval` runs the golden questions (`eval/golden.json`) through each configuration and saves the report to `data/eval/`. Results on the 13-document sample corpus (40 answerable, 5 unanswerable questions; reranker on CPU):

| Configuration | Recall@1 | Recall@5 | MRR | Context recall | Median latency |
|---|---|---|---|---|---|
| dense | 0.90 | 1.00 | 0.95 | 1.00 | 0.13 s |
| sparse | 0.75 | 0.97 | 0.84 | 0.95 | 0.12 s |
| hybrid | 0.82 | 1.00 | 0.90 | 0.97 | 0.12 s |
| hybrid + rerank top 10 | 0.95 | 1.00 | 0.97 | 1.00 | 7.5 s |
| hybrid + rerank top 15, 256 tokens | 0.95 | 1.00 | 0.97 | 1.00 | 9.4 s |
| hybrid + rerank top 30 | 0.95 | 1.00 | 0.97 | 1.00 | 20.7 s |

- **Reranking improves ordering** (Recall@1 0.82 → 0.95), and depth beyond 10 gave no further gain on this set. The default of 20 leaves headroom for the full 1,000-document corpus, where the right chunk may rank lower in search; it was not measured separately (expect ~14 s on CPU).
- **`--max-length 256`** gave the same quality as 512 at the same depth, so 512 stays the default.
- **`--min-score` is off by default.** Rerank scores separate clearly unrelated questions (top score < 0.1) from answerable ones (lowest 0.70), but not near-miss questions whose context is on-topic without the answer (top scores 0.76–0.90). A cut-off would drop good context without catching those; the answer step handles them by refusing when the context lacks the answer.
- Reranker latency is an open item, to be revisited once the full pipeline is built.

## `kb eval-answers` — answer evaluation

| Option | Meaning |
|---|---|
| `--questions LIST` | Only these golden IDs, e.g. `q001,q012` |
| `--types LIST` | Only these question types: `lookup`, `howto`, `compare`, `unanswerable` |
| `--no-judge` | Skip the LLM faithfulness judge (about a third faster) |
| `--provider` | `auto`, `ollama` or `openai` for the answers (the judge always uses Ollama) |
| `--no-compare` | Answer comparisons with one search (baseline for the comparison path) |
| `--details` | Also print missing `must_include` strings and unsupported claims per question |

Runs every golden question through the full pipeline (full access, no release filter) and scores:

- **Status:** answerable questions answered, unanswerable ones refused (by the gate or the LLM).
- **Content:** share of `must_include` strings, article numbers and URLs from the expected answer that appear in the answer.
- **Citations:** precision (cited sources on a golden document and pages) and document recall (golden documents cited, which matters for comparisons); whether the context held a golden source at all, to separate retrieval from generation failures.
- **Faithfulness (LLM judge):** qwen splits the answer into claims and quotes the evidence for each; a claim counts as supported only if its quote occurs word for word in the context. Same model judging its own answers: read flagged claims, don't treat 1.0 as proof.
- **Style flags:** answered on the comparison path (C), no `[n]` markers (M), a not-found sentence removed (N), talk about "the sources" (T).

One line per question as it runs, then a summary; the full report (answers, raw model output, unsupported claims, settings and a prompt hash for comparing runs) goes to `data/eval/answers-<timestamp>.json`. With the judge, expect ~30–40 s per question.

```bash
uv run kb eval-answers --questions q012,q043 --details     # quick check (~1 min)
uv run kb eval-answers --types unanswerable                # refusals only
uv run kb eval-answers --no-judge                          # all 45, without the judge (~20 min)
uv run kb eval-answers                                     # all 45 with the judge (~25-30 min)
```
