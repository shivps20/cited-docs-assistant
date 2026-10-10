# Command reference

Every `kb` command with its options and examples. Run them from the project root; `uv run` uses the project's virtual environment, so no activation is needed.

## Ingestion at a glance

```bash
uv run kb parse [--doc DOC_ID] [--force]   # parse with Docling; results cached in data/parsed/
uv run kb status                           # status and parse statistics per document
uv run kb inspect DOC_ID [--details]       # rebuilt section tree (numbers, pages, size)
uv run kb inspect DOC_ID --section 3.1.7   # content of one section as it will be chunked
uv run kb inspect DOC_ID --chunks [--section 3.1.7]   # chunks as they will be embedded (preview)
uv run kb chunk [--doc DOC_ID]             # build + store sections and chunks in SQLite
uv run kb coverage                         # are golden facts present in the stored chunks?
uv run kb index [--doc DOC_ID] [--force] [--prune]   # embed (bge-m3, GPU) and write to Qdrant
```

`kb index` re-embeds only documents that were (re)chunked; if only manifest values changed (groups, external_ok, release range, latest revision) it updates the Qdrant payload in place. `--prune` removes documents that are no longer in the manifest. Stop any loaded Ollama model first (`ollama stop <model>`): the 6 GB GPU cannot hold both.

Status and statistics are also stored in the `documents` table of `data/kb.db`.

> **Editing in Excel:** Excel rewrites versions such as `1.0` as `1` and `1.10` as `1.1`, which changes which revision counts as latest. Format the `version` column as Text, or edit the CSV in a text editor.

### Infrastructure

**Qdrant (Docker)**

```bash
docker compose -f deploy/docker-compose.yml up -d          # start Qdrant (also restarts with Docker Desktop)
docker compose -f deploy/docker-compose.yml ps             # is it running?
docker compose -f deploy/docker-compose.yml logs -f qdrant # follow the log (Ctrl+C to stop following)
docker compose -f deploy/docker-compose.yml down           # stop; data stays in the named volumes
docker compose -f deploy/docker-compose.yml pull           # fetch the image version pinned in the file
```

Dashboard: <http://127.0.0.1:6444/dashboard>. To change the port, edit the host side of `ports` in `deploy/docker-compose.yml` (`"127.0.0.1:<new>:6333"`), set `QDRANT_URL` in `.env` to match, then `docker compose -f deploy/docker-compose.yml up -d`.

**Ollama**

```bash
ollama pull qwen2.5:7b-instruct   # download the LLM (once)
ollama ps                         # models loaded right now (and whether on GPU)
ollama stop qwen2.5:7b-instruct   # unload it, freeing the GPU before ingestion
```

### Setup scripts

| Command | Options | What it does |
|---|---|---|
| `uv sync` | | Create `.venv` and install dependencies (CUDA torch from the PyTorch index) |
| `uv run python scripts/init_qdrant.py` | `--recreate` delete and recreate the collection (drops all points); `--yes` skip the confirmation | Create the `kb_chunks` collection (dense 1024 cosine + sparse) and its payload indexes; safe to re-run |
| `uv run python scripts/init_db.py` | `--reset` delete and recreate the database; `--yes` skip the confirmation | Create `data/kb.db` or apply pending schema migrations (needed after pulling code with a new migration: the `kb` commands and the server refuse an out-of-date schema) |
| `uv run python scripts/download_models.py` | `--verify-only` skip downloads, only run the smoke tests | Download bge-m3, bge-reranker-v2-m3 and Docling models into `./models`, then verify they load offline |
| `uv run python scripts/check_env.py` | `--deep` also load the LLM and generate a token | Health check: .env, Qdrant, SQLite schema, Ollama, GPU, models, manifest, disk |

```bash
uv run python scripts/check_env.py --deep
uv run python scripts/init_qdrant.py --recreate --yes   # start the index from scratch, then `kb index`
uv run python scripts/init_db.py --reset --yes          # empty database, then parse/chunk/index again
```

### `kb manifest`

| Command | What it does |
|---|---|
| `uv run kb manifest scan [--folder PATH] [--dry-run]` | Draft rows for the files (types in `KB_DOC_TYPES`) in `KB_DOCS_DIR`, or in `--folder` (any folder, read where it is: rows get absolute paths), and its subfolders that are not in the manifest: a count per subfolder, files whose content is already listed (or found twice) skipped and reported, doc ids kept unique across subfolders (same file name in two folders gets the folder name as prefix). `--dry-run` shows what would be drafted and writes nothing |
| `uv run kb manifest validate` | Check every row (columns, categories, releases, dates, duplicate IDs, missing files, two rows with the same family and version); lists all problems with line numbers and how to fix a family clash; counts and lists rows still marked for review |
| `uv run kb manifest backfill [--write]` | Fill what older rows lack: the `added` date (first ingestion in `kb.db`, else the file date) and, for rows without a release, the release from the file name or the PDF's first pages. Shows the changes; `--write` applies them (backup in `data/`) and notes each in the `review` column |

Run `validate` after every edit to `config/manifest.csv`.

**Documents where they already are.** Keep `KB_DOCS_DIR=data/documents` and add any other folder with `uv run kb manifest scan --folder "E:/Docs/New"`: its files are read where they are (the manifest holds their absolute path), so nothing has to be copied. Start with `--dry-run` to see what a folder holds; files whose content is already in the manifest are skipped. `.ppt` / `.doc` are converted with LibreOffice when parsed (once, cached). Drafted rows give access to everyone (`allowed_groups=all`) and keep `external_ok=false`: review access, category, title, version and release range before `kb index`.

### `kb parse` — Docling parsing

| Option | Meaning |
|---|---|
| `--doc DOC_ID` | Only this document; repeatable |
| `--force` | Re-parse even if a cached result exists in `data/parsed/` |

```bash
uv run kb parse                                       # all new or changed documents
uv run kb parse --doc sso-setup --force
```

Parsing typically takes 0.1–0.5 s per page on the GPU (a 214-page guide in about 30 s). Long batches (`parse`, `chunk`, `index`, `coverage`, `eval`, `eval-answers`) keep Windows awake while they run; keep a laptop plugged in with the lid open, because Modern Standby otherwise slows the process down heavily and then sleeps. `kb parse` prints a WARN for any document that took over 2 minutes while using under a tenth of that in CPU time (the machine was asleep or throttled); re-parse such a document with `--doc DOC_ID --force` to record its real time.

### `kb status`

Status per document (parsed / chunked / indexed), parse statistics (pages, seconds, headings, tables, pictures, empty pages) and chunk counts. No options.

### `kb inspect` — review a document's structure and chunks

| Option | Meaning |
|---|---|
| `DOC_ID` | Document to show (required) |
| `--section NUMBER` | Print one section's content, e.g. `3.1.7` |
| `--details` | Also list unmatched TOC entries, demoted headings and removed repeated lines |
| `--chunks` | Show the chunks as they will be embedded (all, or of `--section`) |

```bash
uv run kb inspect install-guide                 # section tree with pages and sizes
uv run kb inspect install-guide --details       # what the structure builder could not place
uv run kb inspect install-guide --section 3.1           # one section's text
uv run kb inspect install-guide --chunks --section 3.1  # that section's chunks
```

### `kb chunk` — build and store sections and chunks

| Option | Meaning |
|---|---|
| `--doc DOC_ID` | Only this document; repeatable |

Always rebuilds from the cached parse and replaces the document's sections and chunks in SQLite; the next `kb index` then re-embeds it.

```bash
uv run kb chunk
uv run kb chunk --doc sso-setup --doc role-overview
```

### `kb coverage` — golden facts present in the chunks?

| Option | Meaning |
|---|---|
| `--all` | List passing questions too, not only failures |

Checks that every `must_include` string of the golden set appears in a stored chunk of the cited document and pages. Expected output: `40/40 answerable golden questions fully covered`.

### `kb index` — embed and write to Qdrant

| Option | Meaning |
|---|---|
| `--doc DOC_ID` | Only this document; repeatable |
| `--force` | Re-embed even if already indexed |
| `--prune` | Delete points and rows of documents no longer in the manifest |

Re-embeds only (re)chunked documents; manifest-only changes update the Qdrant payload in place. Uses the GPU: run `ollama stop <model>` first.

```bash
uv run kb index
uv run kb index --doc install-guide --force
uv run kb index --prune          # after removing rows from config/manifest.csv
```

### `kb search` — retrieval without the LLM

| Option | Default | Meaning |
|---|---|---|
| `QUERY` | | The question (required, in quotes) |
| `--groups GROUP` | `all` | User access group(s); repeatable or comma-separated |
| `--release RELEASE` | any | Only documents that apply to this release, e.g. `R2024x` |
| `--mode` | `hybrid` | `hybrid`, `dense` or `sparse` |
| `--no-rerank` | | Skip the cross-encoder (fast; search order only) |
| `--candidates N` | 30 | Chunks retrieved before reranking |
| `--rerank-top N` | `KB_RERANK_TOP` (20) | Rerank only the first N candidates; `0` = all |
| `--max-length N` | 512 | Reranker input length in tokens |
| `--min-score X` | off | Drop context chunks with rerank score below X |
| `--show N` | 10 | Ranked results to print |
| `--context` | | Print the full assembled context the LLM would receive |

```bash
uv run kb search "Which port does the application server use?"
uv run kb search "Which port does the application server use?" --mode dense --no-rerank   # fast, no reranker
uv run kb search "Which tool captures traffic for a performance analysis?"                       # restricted doc hidden
uv run kb search "Which tool captures traffic for a performance analysis?" --groups internal  # visible with access
uv run kb search "Which SQL Server version is supported?" --release R2025x      # release filter
uv run kb search "How do I enable single sign-on?" --context --show 5         # see what the LLM gets
uv run kb search "..." --rerank-top 0 --min-score 0.3                           # rerank all, drop weak context
```

### `kb models` — model catalogue

| Command | What it does |
|---|---|
| `uv run kb models list` | Validate `config/models.yaml` (every problem listed at once) and show each model: adapter, local / external, whether its key is set, context and output size, model id and the roles it has |
| `uv run kb models check [--model NAME]` | Send every configured model (or the named ones) a tiny text prompt and a JSON prompt: reachable, seconds, token counts, whether JSON works. Models without a key are skipped; costs a few tokens per model |

Without `config/models.yaml` it shows the catalogue built from the older settings (`LLM_MODEL`, `OPENAI_*`).

### `kb cache` — answer cache

| Command | What it does |
|---|---|
| `uv run kb cache stats` | Entries (of the current corpus and stale), answers served from the cache, the most-served questions |
| `uv run kb cache clear` | Remove every entry |
| `uv run kb cache clear --stale` | Remove only entries of an older corpus (they can no longer be hit; storing a new answer also removes them) |

### `kb ask` — answer a question with citations

| Option | Default | Meaning |
|---|---|---|
| `QUERY` | | The question (required, in quotes) |
| `--groups GROUP` | `all` | User access group(s); repeatable or comma-separated |
| `--release RELEASE` | any | Only documents that apply to this release, e.g. `R2024x` |
| `--model NAME` | the answer role in `config/models.yaml` | Answer model by catalogue name (`auto`, `ollama`, `openai` still accepted). External models only see documents with `external_ok`; otherwise the local fallback answers, with a notice |
| `--mode` | `hybrid` | `hybrid`, `dense` or `sparse` |
| `--no-rerank` | | Skip the reranker; without a rerank score the gate always passes |
| `--rerank-top N` | `KB_RERANK_TOP` (20) | Rerank only the first N candidates; `0` = all |
| `--show-context` | | Print the context sent to the LLM before the answer |
| `--no-stream` | | Print the answer only when it is complete |
| `--no-compare` | | Answer comparison questions with one search instead of one search per side |
| `--no-refusal-retry` | `KB_REFUSAL_RETRY` (on) | No second attempt with the best-matching sources when the model finds no answer |
| `--no-cache` | `KB_ANSWER_CACHE` (on) | Neither read nor store the answer cache: always search and ask the model |

After the answer it prints the sources, any notes (removed citations, provider choice, a comparison answered with one search, an answer found on the second attempt), the status (`answered`, or `not found` by the gate or by the LLM), timings and the trace ID.

```bash
uv run kb ask "How do I enable single sign-on?"
uv run kb ask "Which port does the application server use?" --show-context              # see what the LLM was given
uv run kb ask "Which tool captures traffic for a performance analysis?" --groups internal   # restricted document
uv run kb ask "Which SQL Server version is supported?" --release R2026x
uv run kb ask "How do I configure NGINX as a reverse proxy?"                   # expect "not found"
uv run kb ask "How does the database setup differ between MSSQL and Oracle?"   # comparison: one search per side
uv run kb ask "..." --model claude-opus                                      # a model from config/models.yaml (key in .env)
```

### `kb serve` — chat API

Starts the chat API and web UI; endpoints, the chat turn and the UI are described in [api.md](api.md).

| Option | Default | Meaning |
|---|---|---|
| `--host` | `KB_API_HOST` (127.0.0.1) | Interface to listen on; keep the loopback address (users are not authenticated) |
| `--port` | `KB_API_PORT` (8000) | Port |

```bash
uv run kb serve                 # http://127.0.0.1:8000 ; models load once (~20 s)
uv run kb serve --port 8012     # a second instance, e.g. while another one runs
```

### `kb eval`, `kb eval-answers` — evaluation

Options, metrics and results are described in [evaluation.md](evaluation.md).

```bash
uv run kb eval --configs dense,hybrid+rr10 --misses        # retrieval only (~7 min with the reranker)
uv run kb eval-answers --questions q012,q043 --details     # a few questions with the judge (~1 min)
uv run kb eval-answers --types compare                     # one question type
uv run kb eval-answers --no-judge                          # whole set without the judge (~10 s per question)
uv run kb eval-answers                                     # whole set with the judge (~1 h on the GPU)
uv run kb eval-answers --model claude-opus --judge-model local-qwen   # another answer model, same judge
```

Reports go to `data/eval/` (`retrieval-*.json`, `answers-*.json`).

### `kb calibrate` — the "not found" threshold

Replays candidate gate thresholds (`KB_NOT_FOUND_SCORE`) on the latest full `kb eval-answers` report per answer model, without running any model, and recommends one. Details in [evaluation.md](evaluation.md#kb-calibrate--the-not-found-threshold).

| Option | Default | Meaning |
|---|---|---|
| `--report PATH` | latest full run per answer model | Answer evaluation report(s) to use; repeatable |
| `--thresholds LIST` | 0 to 0.9 | Candidate thresholds, comma-separated |
| `--all` | reviewed questions only | Count every golden question |
| `--details` | | List the questions lost or unmeasured per threshold |

```bash
uv run kb calibrate                                   # after a full kb eval-answers
uv run kb calibrate --all --details                   # every question, with the questions behind each number
uv run kb calibrate --report data/eval/answers-<timestamp>.json --thresholds 0.05,0.1,0.15
```

The output ends with a recommended threshold, the refusals that are generation problems rather than the gate's, the number of real questions the gate refuses, and the 👎 reasons from the chat; the result is also saved as `data/eval/calibration-<timestamp>.json`. To change the threshold, set `KB_NOT_FOUND_SCORE` in `.env` and run `kb eval-answers` again.

### Typical workflows

**New or changed document**

```bash
uv run kb manifest scan          # new file -> draft row; fill it in
uv run kb manifest validate
uv run kb parse
uv run kb inspect <doc_id> --details
uv run kb chunk --doc <doc_id>
ollama stop qwen2.5:7b-instruct
uv run kb index
uv run kb coverage
```

**After changing the structure or chunking code:** `uv run kb chunk` (all, or the affected `--doc`s), then `uv run kb index`, `uv run kb coverage` and `uv run kb eval --configs dense,hybrid+rr10`.

**Only manifest values changed** (groups, external_ok, release range, version): `uv run kb manifest validate`, then `uv run kb index` updates the payloads without re-embedding.

Either way the corpus changes, so earlier cached answers are no longer served; remove them with `uv run kb cache clear --stale` (storing the next answer also does it).

**After pulling new code**

```bash
uv sync                                    # new or updated dependencies
uv run python scripts/init_db.py           # apply new schema migrations (e.g. v6: feedback reasons)
uv run pytest -q
uv run python scripts/check_env.py
```

**Changing the language models** (`config/models.yaml`)

```bash
uv run kb models list                      # validate: every problem listed at once
uv run kb models check --model claude-opus # reachable, speed, JSON (costs a few tokens)
uv run kb ask "..." --model claude-opus    # try it on one question
```

A running `kb serve` picks up the changed file without a restart; cached answers of the old settings are no longer served.

**Measuring a change in answers** (prompt, retrieval settings, answer model)

```bash
uv run kb eval-answers --types compare     # the affected type first
uv run kb eval-answers                     # the whole set before closing a phase
uv run kb calibrate                        # is the "not found" threshold still right?
```

Compare the summary with the previous report in `data/eval/`; every report records the models, settings and a hash of the prompt.

**After editing the golden set** (`eval/golden.json`)

```bash
uv run python eval/check_golden.py         # must_include in the expected answers, pages and sources exist
uv run kb coverage                         # the key facts are in the stored chunks
```

Mark reviewed questions with `"status": "reviewed"`: `kb calibrate` counts only those by default.

### Development and checks

```bash
uv run pytest -q                          # test suite, quiet output
uv run pytest tests/ingest/test_structure.py -q   # one test file (tests mirror src/kb)
uv run pytest -k cache -q                 # tests whose name contains "cache"
uv run ruff check src tests scripts eval  # lint
uv run ruff check --fix src tests scripts eval   # lint and apply safe fixes
uv run python eval/check_golden.py        # golden set consistency, after editing eval/golden.json
```

### Inspecting the data

- **Qdrant:** dashboard at <http://127.0.0.1:6444/dashboard> → Collections → `kb_chunks`.
- **SQLite:** open `data/kb.db` read-only in DB Browser for SQLite. Useful queries:

```sql
SELECT doc_id, status, page_count, chunk_count, parse_seconds, embed_seconds FROM documents;
SELECT section_id, heading_path, page_start, page_end, token_count FROM sections WHERE doc_id = 'install-guide';
SELECT trace_id, route, query, top_rerank_score, gate_decision, llm_model, total_ms FROM traces ORDER BY created_at DESC LIMIT 20;
SELECT query, answer, sources FROM traces WHERE route = 'answer' ORDER BY created_at DESC LIMIT 5;
SELECT stage, duration_ms, data FROM trace_stages WHERE trace_id = '<trace_id>' ORDER BY seq;
SELECT f.rating, f.reason, f.comment, t.query, t.top_rerank_score, t.gate_decision FROM feedback f JOIN traces t USING (trace_id);
SELECT query_norm, user_groups, release_filter, hit_count, created_at, last_hit_at FROM answer_cache ORDER BY hit_count DESC;
SELECT query, cache_hit, total_ms FROM traces WHERE cache_hit = 1 ORDER BY created_at DESC LIMIT 10;
```

### Local project notes (not in the repository)

The design document, plan, trade-offs, technical debt and code-flow pages live in `docs/local/`, which is git-ignored because they use real document names and questions. The design document is written in Markdown (`docs/local/DESIGN.md`) and also built as one offline HTML page with the diagrams drawn in (`docs/local/DESIGN.html`):

```bash
uv run python docs/local/design_build/build_design_html.py   # rebuild DESIGN.html after editing DESIGN.md
start docs/local/DESIGN.html                                 # open it (Windows; macOS: open, Linux: xdg-open)
```

After changing or adding a diagram in `DESIGN.md`, render the diagrams again before rebuilding:

```bash
uv run python docs/local/design_build/build_design_html.py   # also writes the diagram sources (blocks.json)
python docs/local/design_build/save_server.py                # serves the build folder on http://127.0.0.1:8766
```

Open <http://127.0.0.1:8766/render.html>, click **Render all** (it loads Mermaid from a CDN, only at this step), stop the server with Ctrl+C, and run the build script once more. Never commit or publish `DESIGN.md` or `DESIGN.html`.

### Git workflow (one branch per phase)

```bash
git checkout -b phase-3-answers                        # start a phase
git add <files>                                        # stage explicit paths
git commit -m "Phase 3 step 1: ..."
git push -u origin phase-3-answers                     # first push of the branch; later just `git push`
git checkout master                                    # finish the phase
git pull
git merge --no-ff phase-3-answers -m "Merge Phase 3: answer generation"
git push
```
