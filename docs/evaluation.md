# Evaluation

How retrieval and answers are measured against the golden question set (`eval/golden.json`, local; fictional example in [eval/golden.example.json](../eval/golden.example.json)). Ingestion coverage is checked with `kb coverage` (see [commands.md](commands.md)).

## `kb eval` — retrieval evaluation

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--configs LIST</code></td><td>Comma-separated subset of <code>dense</code>, <code>sparse</code>, <code>hybrid</code>, <code>hybrid+rr10</code>, <code>hybrid+rr15@256</code>, <code>hybrid+rr30</code> (default: all; the reranked ones take minutes)</td></tr>
<tr><td><code>--questions LIST</code></td><td>Only these golden IDs, e.g. <code>q001,q012</code></td></tr>
<tr><td><code>--misses</code></td><td>List questions without a hit in the top 5</td></tr>
</tbody>
</table>

Prints Recall@1/5/10, MRR, context recall, all-docs@5 for comparisons, latency, scores for answerable vs unanswerable questions, and the access and release filter checks. The full report goes to `data/eval/retrieval-<timestamp>.json`.

```bash
uv run kb eval --configs dense,hybrid                 # fast (~10 s)
uv run kb eval --configs dense,hybrid+rr10 --misses   # with the reranker (~7 min)
uv run kb eval --questions q022,q040 --configs hybrid+rr10
uv run kb eval                                        # all six configurations (~20 min)
```

### Results behind the retrieval defaults

`uv run kb eval` runs the golden questions (`eval/golden.json`) through each configuration and saves the report to `data/eval/`. Results on the 13-document sample corpus (40 answerable, 5 unanswerable questions; reranker on CPU):

<table style="width:100%">
<colgroup><col style="width:39%"><col style="width:11%"><col style="width:11%"><col style="width:9%"><col style="width:13%"><col style="width:17%"></colgroup>
<thead><tr><th>Configuration</th><th>Recall@1</th><th>Recall@5</th><th>MRR</th><th>Context recall</th><th>Median latency</th></tr></thead>
<tbody>
<tr><td>dense</td><td>0.90</td><td>1.00</td><td>0.95</td><td>1.00</td><td>0.13 s</td></tr>
<tr><td>sparse</td><td>0.75</td><td>0.97</td><td>0.84</td><td>0.95</td><td>0.12 s</td></tr>
<tr><td>hybrid</td><td>0.82</td><td>1.00</td><td>0.90</td><td>0.97</td><td>0.12 s</td></tr>
<tr><td>hybrid + rerank top 10</td><td>0.95</td><td>1.00</td><td>0.97</td><td>1.00</td><td>7.5 s</td></tr>
<tr><td>hybrid + rerank top 15, 256 tokens</td><td>0.95</td><td>1.00</td><td>0.97</td><td>1.00</td><td>9.4 s</td></tr>
<tr><td>hybrid + rerank top 30</td><td>0.95</td><td>1.00</td><td>0.97</td><td>1.00</td><td>20.7 s</td></tr>
</tbody>
</table>

- **Reranking improves ordering** (Recall@1 0.82 → 0.95), and depth beyond 10 gave no further gain on this set. The default of 20 leaves headroom for the full 1,000-document corpus, where the right chunk may rank lower in search; it was not measured separately (expect ~14 s on CPU).
- **`--max-length 256`** gave the same quality as 512 at the same depth, so 512 stays the default.
- **`--min-score` is off by default.** Rerank scores separate clearly unrelated questions (top score < 0.1) from answerable ones (lowest 0.70), but not near-miss questions whose context is on-topic without the answer (top scores 0.76–0.90). A cut-off would drop good context without catching those; the answer step handles them by refusing when the context lacks the answer.
- Reranker latency is an open item, to be revisited once the full pipeline is built.

## `kb eval-answers` — answer evaluation

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--questions LIST</code></td><td>Only these golden IDs, e.g. <code>q001,q012</code></td></tr>
<tr><td><code>--types LIST</code></td><td>Only these question types: <code>lookup</code>, <code>howto</code>, <code>compare</code>, <code>unanswerable</code></td></tr>
<tr><td><code>--no-judge</code></td><td>Skip the LLM faithfulness judge (about a third faster)</td></tr>
<tr><td><code>--model NAME</code></td><td>Answer model from <code>config/models.yaml</code> (default: the answer role); the report records it as <code>answer_model</code></td></tr>
<tr><td><code>--judge-model NAME</code></td><td>Faithfulness judge model (default: the judge role, local by default)</td></tr>
<tr><td><code>--no-compare</code></td><td>Answer comparisons with one search (baseline for the comparison path)</td></tr>
<tr><td><code>--no-refusal-retry</code></td><td>No second attempt after a refusal (baseline for the refusal retry, which is on by default)</td></tr>
<tr><td><code>--details</code></td><td>Also print missing <code>must_include</code> strings and unsupported claims per question</td></tr>
</tbody>
</table>

Runs every golden question through the full pipeline (full access, no release filter) and scores:

- **Status:** answerable questions answered, unanswerable ones refused (by the gate or the LLM).
- **Content:** share of `must_include` strings, article numbers and URLs from the expected answer that appear in the answer.
- **Citations:** precision (cited sources on a golden document and pages) and document recall (golden documents cited, which matters for comparisons); whether the context held a golden source at all, to separate retrieval from generation failures.
- **Faithfulness (LLM judge):** qwen splits the answer into claims and quotes the evidence for each; a claim counts as supported only if its quote occurs word for word in the context (a quote of several lines, e.g. the same fact quoted from two sources, and a quote shortened with "..." are checked piece by piece: every piece must occur word for word). A claim without a usable quote still counts when every distinctive value in it (error codes, numbers of 3+ digits, file names, paths, `inline code`) is in the context; the summary shows how many claims were supported this way. Answers over 1,200 characters are judged in parts (split between paragraphs), with a judge output cap of 3,000 tokens. Same model judging its own answers: read flagged claims, don't treat 1.0 as proof. Since the piece-by-piece check of multi-line quotes, scores are somewhat higher than in earlier reports for the same answers: compare runs scored with the same code. A single answer can be judged on demand with `kb trace <id> --faithfulness` or **Check faithfulness** in the chat's Behind the scenes panel.
- **Command check (no LLM):** every line of a code block and every `inline code` span in the answer must appear in the context, comparing letters and digits only (case, spacing, punctuation and prompt styles such as `MQL>` / `MQL >` are ignored; a changed value is not). It catches a changed value in a command, such as `octreedepth 6` where the source says 5. Reported in the summary; `--details` lists the commands that were not found.
- **Style flags:** answered on the comparison path (C), no `[n]` markers (M), a not-found sentence removed (N), talk about "the sources" (T), answered only on the retry with fewer sources (R; the summary lists them), sections added by the comparison read step (S, only for a model with `compare_read: true`).

One line per question as it runs, then a summary; the full report (answers, raw model output, unsupported claims, settings and a prompt hash for comparing runs) goes to `data/eval/answers-<timestamp>.json`. With the judge, expect ~30–40 s per question.

```bash
uv run kb eval-answers --questions q012,q043 --details     # quick check (~1 min)
uv run kb eval-answers --types unanswerable                # refusals only
uv run kb eval-answers --no-judge                          # whole set, without the judge (~10 s per question)
uv run kb eval-answers                                     # whole set with the judge (~25-30 s per question)
```

## `kb calibrate` — the "not found" threshold

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--report PATH</code></td><td>Answer evaluation report(s) to use, repeatable (default: the latest full run per answer model in <code>data/eval/</code>)</td></tr>
<tr><td><code>--thresholds LIST</code></td><td>Candidate thresholds, comma-separated (default: 0 to 0.9)</td></tr>
<tr><td><code>--all</code></td><td>Count every golden question, not only those marked <code>status: reviewed</code></td></tr>
<tr><td><code>--details</code></td><td>List, per threshold, the questions whose answers would be lost or whose outcome is unmeasured</td></tr>
</tbody>
</table>

The gate refuses without calling the LLM when the top rerank score is below `KB_NOT_FOUND_SCORE`. `kb calibrate` replays every candidate threshold on an evaluation report without running any model: a question scoring below the candidate is refused by the gate, every other question keeps the outcome the report recorded. That is exact for thresholds at or above the one the report was run with; below it, questions the gate refused at run time would reach the LLM with an unknown outcome ("unmeasured"). Answers are re-scored against the current golden set, so a report from before a golden-set review still counts correctly.

Per threshold it shows answers lost (and how many of them had every key fact), wrong refusals, unanswerable questions refused (and how many by the gate), unanswerable questions let through, LLM calls saved, answers the refusal retry rescued that still reach the LLM, and unmeasured questions. It then recommends the midpoint between the highest-scoring unanswerable question the gate can catch and the lowest-scoring answered question (the widest margin that loses no answer), lists refusals that happened despite a high score (a generation problem, not the gate's), and counts real questions and feedback from the traces: 👎 reasons are counted, a "should have answered" on a gate refusal just below the threshold is reported as evidence for a lower one (far below it, as a matter of access or coverage), and a "should have refused" on an answer is placed below or above the threshold. The result is also saved as `data/eval/calibration-<timestamp>.json`.

```bash
uv run kb calibrate                      # reviewed questions, latest full run per answer model
uv run kb calibrate --all --details      # every question, with the questions behind each number
```

Result on the 221-document corpus (2026-10-10, 30 reviewed questions): keep 0.10. It sits midway between the highest-scoring unanswerable question the gate can catch (0.053) and the lowest-scoring answered one (0.147, a broad question); 10 of the 12 unanswerable questions score as high as answered ones, so only the model's refusal can catch them. Re-run `kb calibrate` after `kb eval-answers` with a new answer model.

