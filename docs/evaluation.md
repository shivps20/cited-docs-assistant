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
| `--model NAME` | Answer model from `config/models.yaml` (default: the answer role); the report records it as `answer_model` |
| `--judge-model NAME` | Faithfulness judge model (default: the judge role, local by default) |
| `--no-compare` | Answer comparisons with one search (baseline for the comparison path) |
| `--no-refusal-retry` | No second attempt after a refusal (baseline for the refusal retry, which is on by default) |
| `--details` | Also print missing `must_include` strings and unsupported claims per question |

Runs every golden question through the full pipeline (full access, no release filter) and scores:

- **Status:** answerable questions answered, unanswerable ones refused (by the gate or the LLM).
- **Content:** share of `must_include` strings, article numbers and URLs from the expected answer that appear in the answer.
- **Citations:** precision (cited sources on a golden document and pages) and document recall (golden documents cited, which matters for comparisons); whether the context held a golden source at all, to separate retrieval from generation failures.
- **Faithfulness (LLM judge):** qwen splits the answer into claims and quotes the evidence for each; a claim counts as supported only if its quote occurs word for word in the context. A claim without a usable quote still counts when every distinctive value in it (error codes, numbers of 3+ digits, file names, paths, `inline code`) is in the context; the summary shows how many claims were supported this way. Answers over 1,200 characters are judged in parts (split between paragraphs), with a judge output cap of 3,000 tokens. Same model judging its own answers: read flagged claims, don't treat 1.0 as proof.
- **Command check (no LLM):** every line of a code block and every `inline code` span in the answer must appear word for word in the context (case, spacing and markdown ignored). It catches a changed value in a command, such as `octreedepth 6` where the source says 5. Reported in the summary; `--details` lists the commands that were not found.
- **Style flags:** answered on the comparison path (C), no `[n]` markers (M), a not-found sentence removed (N), talk about "the sources" (T), answered only on the retry with fewer sources (R; the summary lists them), sections added by the comparison read step (S, only for a model with `compare_read: true`).

One line per question as it runs, then a summary; the full report (answers, raw model output, unsupported claims, settings and a prompt hash for comparing runs) goes to `data/eval/answers-<timestamp>.json`. With the judge, expect ~30–40 s per question.

```bash
uv run kb eval-answers --questions q012,q043 --details     # quick check (~1 min)
uv run kb eval-answers --types unanswerable                # refusals only
uv run kb eval-answers --no-judge                          # whole set, without the judge (~10 s per question)
uv run kb eval-answers                                     # whole set with the judge (~25-30 s per question)
```

## `kb calibrate` — the "not found" threshold

| Option | Meaning |
|---|---|
| `--report PATH` | Answer evaluation report(s) to use, repeatable (default: the latest full run per answer model in `data/eval/`) |
| `--thresholds LIST` | Candidate thresholds, comma-separated (default: 0 to 0.9) |
| `--all` | Count every golden question, not only those marked `status: reviewed` |
| `--details` | List, per threshold, the questions whose answers would be lost or whose outcome is unmeasured |

The gate refuses without calling the LLM when the top rerank score is below `KB_NOT_FOUND_SCORE`. `kb calibrate` replays every candidate threshold on an evaluation report without running any model: a question scoring below the candidate is refused by the gate, every other question keeps the outcome the report recorded. That is exact for thresholds at or above the one the report was run with; below it, questions the gate refused at run time would reach the LLM with an unknown outcome ("unmeasured"). Answers are re-scored against the current golden set, so a report from before a golden-set review still counts correctly.

Per threshold it shows answers lost (and how many of them had every key fact), wrong refusals, unanswerable questions refused (and how many by the gate), unanswerable questions let through, LLM calls saved, answers the refusal retry rescued that still reach the LLM, and unmeasured questions. It then recommends the midpoint between the highest-scoring unanswerable question the gate can catch and the lowest-scoring answered question (the widest margin that loses no answer), lists refusals that happened despite a high score (a generation problem, not the gate's), and counts real questions and feedback from the traces: 👎 reasons are counted, a "should have answered" on a gate refusal just below the threshold is reported as evidence for a lower one (far below it, as a matter of access or coverage), and a "should have refused" on an answer is placed below or above the threshold. The result is also saved as `data/eval/calibration-<timestamp>.json`.

```bash
uv run kb calibrate                      # reviewed questions, latest full run per answer model
uv run kb calibrate --all --details      # every question, with the questions behind each number
```

Result on the 221-document corpus (2026-10-10, 30 reviewed questions): keep 0.10. It sits midway between the highest-scoring unanswerable question the gate can catch (0.053) and the lowest-scoring answered one (0.147, a broad question); 10 of the 12 unanswerable questions score as high as answered ones, so only the model's refusal can catch them. Re-run `kb calibrate` after `kb eval-answers` with a new answer model.

