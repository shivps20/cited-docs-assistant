# Chat API and web UI

`kb serve` runs a FastAPI server with the chat UI, the chat endpoint (server-sent events), sessions, feedback and health checks.

## `kb serve`

| Option | Default | Meaning |
|---|---|---|
| `--host` | `KB_API_HOST` (127.0.0.1) | Interface to listen on. Keep the loopback address: users are not authenticated |
| `--port` | `KB_API_PORT` (8000) | Port |

Loads bge-m3, the reranker and the LLM providers **once** at startup (~20 s), so each request costs only retrieval and generation. **Chat UI: <http://127.0.0.1:8000>**; interactive API docs: <http://127.0.0.1:8000/docs>.

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Status of SQLite, Qdrant (point count), Ollama (model pulled) and the loaded models; `status` is `ok` or `degraded` |
| `GET /api/users` | Users from `config/users.yaml`, for the UI's user picker |
| `GET /api/models` | Models from `config/models.yaml` for the UI's model menu: `default` (the answer role), `fallback`, and per model `name`, `model`, `adapter`, `location` (local / external) and `ready` (key set) |
| `GET /api/me` | The current user and the groups their searches are filtered by |
| `POST /api/sessions` | Start a conversation |
| `GET /api/sessions` | The current user's conversations, most recent first, titled by their first question |
| `GET /api/sessions/{id}` | One conversation with its messages (404 if it is another user's) |
| `PUT /api/sessions/{id}/release` | Set (`{"release": "R2025x"}`) or clear (`{"release": null}`) the conversation's release filter |
| `POST /api/feedback` | Rate one of your answers: `{"trace_id": "…", "rating": 1 \| -1, "comment": "…" (optional)}`; rating again replaces the earlier rating (404 for another user's answer). A 👎 also removes the answer from the answer cache (whether it was the stored answer or a cached copy), so the next ask is answered afresh |
| `GET /` | The chat UI |
| `POST /api/chat` | Ask a question: body `{"question": "…", "session_id": "…" (optional: a new conversation without it), "model": "claude-opus" (optional: a name from `config/models.yaml`; without it the answer role's model; `provider` is accepted as the older name), "cache": false (optional: answer afresh instead of from the answer cache)}`. An unknown model is a 400 listing the configured ones. Answers as a stream of server-sent events (below); `final` names the model that answered (`model_profile`) |

The user is named by the `X-KB-User` header (the default user from `config/users.yaml` when absent); unknown users get 403. Access groups always come from `config/users.yaml` (example: [config/users.example.yaml](../config/users.example.yaml)), never from the request, so a client cannot claim more access than configured.

```bash
uv run kb serve
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/sessions -H "X-KB-User: internal_user"
curl http://127.0.0.1:8000/api/sessions -H "X-KB-User: internal_user"
```

In PowerShell, `curl` is an alias for `Invoke-WebRequest` (different options, and a script-execution prompt): type `curl.exe` instead. Or use the interactive docs at <http://127.0.0.1:8000/docs>, where each endpoint has a field for the `X-KB-User` header.

**Chat UI** (<http://127.0.0.1:8000>, one page served by the API, works offline):

- **Model menu** (top right): the usable models from `config/models.yaml` (key set); external ones are marked ↗ and only ever see documents cleared for external use. The choice is remembered in the browser and sent with each question; each answer's footer names the model that wrote it.
- **User picker** (top right): users from `config/users.yaml`, with their groups; remembered in the browser. Switching user starts a new conversation.
- **Conversations** (left): your conversations, newest first; click one to reopen it with its answers, sources and ratings.
- **Asking:** Enter sends, Shift+Enter adds a line. Progress shows "Understanding your follow-up…", "Searching the documents (R2026x)…", "Writing the answer…"; the answer streams, then is replaced by the cleaned version.
- **Each answer shows:** "Interpreted as …" for a condensed follow-up and the release used; the answer with clickable `[n]` (highlights the source); the articles and links block; Sources; notices; a collapsible "Context sent to the LLM"; retrieval / generation times, model, tokens/s and the trace id; 👍 / 👎 with an optional comment.
- **Release chip** (above the input): the conversation's release; ✕ clears it. Name a release in a question to set it.
- **Not found** answers are shown in amber, with whether the gate (no relevant documents) or the LLM (documents do not contain the answer) refused.
- **Health dot** (top): green when SQLite, Qdrant, Ollama and the models are all fine; hover for details.

**Chat turn** (`POST /api/chat`), in order:

1. **Release:** a release in the question (`R2025x`) becomes the conversation's filter and stays for later questions; "any release" / "all releases" clears it; a question naming two releases (a comparison) is not filtered.
2. **Follow-ups:** a question that depends on earlier turns ("and on Oracle?", "how do I change it?") is rewritten by the local LLM into a standalone question from the last 6 messages. Self-contained questions skip this. Condensing always runs locally (the conversation may contain text not cleared for external LLMs); a bad rewrite falls back to the original question.
3. **Answer:** the same pipeline as `kb ask` (retrieval with the user's groups and the release, gate, LLM, citations, article numbers / URLs).
4. **Stored:** the question (with its standalone version) and the answer (with its trace id) in `messages`; the trace keeps the original and the standalone question and a `condense` stage.

Server-sent events, in order: `session` (session id, sticky release) → `status` (`queued` while another turn is running, `condensing`, then `searching` with the standalone question and release; for a comparison also `comparing` while the question is split, then `searching_side` with `side`, `index` and `total` before each side's search (`reading` with the `sections` added by the comparison read step, only when the answer model has `compare_read: true`); `retrying` with `units` when the model found no answer and is asked again with fewer sources, followed by a second `context` event and more `token` events) → `context` (the numbered sources sent to the LLM; `same_text` lists near-identical copies in other documents that were not sent again) → `token` … (the raw answer as it streams) → `final` (the **cleaned** answer, `route` and the compared `sides`, sources (each with its `same_text` copies), article references, notices, gate verdict, provider, timings, trace id), or `error` instead of `final`. A UI should replace the streamed text with `final.answer`. Only one turn runs at a time (shared models and GPU); a request that has to wait gets `status: queued` first and starts when the running turn ends. If the browser disconnects mid-answer, the turn still finishes and is stored: reopening the conversation shows it.

Without the UI: open <http://127.0.0.1:8000/docs>, expand **POST /api/chat**, *Try it out*, and send `{"question": "Which port does the application server use on R2026x?"}` (the docs page shows the events when the answer is complete), then the same with `"session_id"` from the `session` event and `"question": "and on Oracle?"`.
