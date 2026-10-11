# Chat API and web UI

`kb serve` runs a FastAPI server with the chat UI, the chat endpoint (server-sent events), sessions, feedback and health checks.

## `kb serve`

<table style="width:100%">
<colgroup><col style="width:12%"><col style="width:29%"><col style="width:59%"></colgroup>
<thead><tr><th>Option</th><th>Default</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--host</code></td><td><code>KB_API_HOST</code> (127.0.0.1)</td><td>Interface to listen on. Keep the loopback address: users are not authenticated</td></tr>
<tr><td><code>--port</code></td><td><code>KB_API_PORT</code> (8000)</td><td>Port</td></tr>
</tbody>
</table>

Loads bge-m3, the reranker and the LLM providers **once** at startup (~20 s), so each request costs only retrieval and generation. **Chat UI: <http://127.0.0.1:8000>**; interactive API docs: <http://127.0.0.1:8000/docs>.

<table style="width:100%">
<colgroup><col style="width:20%"><col style="width:80%"></colgroup>
<thead><tr><th>Endpoint</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>GET /api/health</code></td><td>Status of SQLite, Qdrant (point count), Ollama (model pulled) and the loaded models; <code>status</code> is <code>ok</code> or <code>degraded</code></td></tr>
<tr><td><code>GET /api/users</code></td><td>Users from <code>config/users.yaml</code>, for the UI's user picker</td></tr>
<tr><td><code>GET /api/models</code></td><td>Models from <code>config/models.yaml</code> for the UI's model menu: <code>default</code> (the answer role), <code>fallback</code>, and per model <code>name</code>, <code>model</code>, <code>adapter</code>, <code>location</code> (local / external) and <code>ready</code> (key set)</td></tr>
<tr><td><code>GET /api/me</code></td><td>The current user and the groups their searches are filtered by</td></tr>
<tr><td><code>POST /api/sessions</code></td><td>Start a conversation</td></tr>
<tr><td><code>GET /api/sessions</code></td><td>The current user's conversations, most recent first, titled by their first question</td></tr>
<tr><td><code>GET /api/sessions/{id}</code></td><td>One conversation with its messages (404 if it is another user's)</td></tr>
<tr><td><code>PUT /api/sessions/{id}/release</code></td><td>Set (<code>{&quot;release&quot;: &quot;R2025x&quot;}</code>) or clear (<code>{&quot;release&quot;: null}</code>) the conversation's release filter</td></tr>
<tr><td><code>POST /api/feedback</code></td><td>Rate one of your answers: <code>{&quot;trace_id&quot;: &quot;…&quot;, &quot;rating&quot;: 1 | -1, &quot;reason&quot;: &quot;…&quot; (optional, only with -1: </code>wrong<code>, </code>incomplete<code>, </code>should_have_answered<code>, </code>should_have_refused<code>), &quot;comment&quot;: &quot;…&quot; (optional)}</code>; a reason with a 👍 or an unknown reason is a 422, and a 👍 clears an earlier reason; rating again replaces the earlier rating (404 for another user's answer). A 👎 also removes the answer from the answer cache (whether it was the stored answer or a cached copy), so the next ask is answered afresh</td></tr>
<tr><td><code>GET /api/traces/{trace_id}</code></td><td>Behind the scenes of one of your answers (404 for another user's): question and search filters; every search candidate with its retrieval (fusion) score and rank, its rerank score and rank (the reranked top 20) and whether it reached the context; the gate decision, top score and threshold; the context units sent; each generation's model, tokens, speed, fallbacks and the exact messages (rebuilt from the stored context, with <code>prompt_identical</code> saying whether today's prompt code still produces the same messages); citations kept and removed, the refusal retry, the command check; a stored faithfulness verdict; the stage timings. An answer served from the cache points to the original trace (<code>cache.served_from</code>)</td></tr>
<tr><td><code>POST /api/traces/{trace_id}/faithfulness[?force=true]</code></td><td>Run the faithfulness judge on one of your answers against the context it was given, on demand (one more LLM call, 20–60 s; it waits for a running chat turn). The verdict (score, claims, quotes, judge model) is stored with the trace and returned; asking again returns it unless <code>force=true</code>. The judge is the catalogue's judge role, or the local model when a source may not leave the machine (<code>external_ok</code>). 409 when there is nothing to judge (a refusal, a cached copy, an answer traced before contexts were stored)</td></tr>
<tr><td><code>GET /</code></td><td>The chat UI</td></tr>
<tr><td><code>POST /api/chat</code></td><td>Ask a question: body <code>{&quot;question&quot;: &quot;…&quot;, &quot;session_id&quot;: &quot;…&quot; (optional: a new conversation without it), &quot;model&quot;: &quot;claude-opus&quot; (optional: a name from </code>config/models.yaml<code>; without it the answer role's model; </code>provider<code> is accepted as the older name), &quot;cache&quot;: false (optional: answer afresh instead of from the answer cache)}</code>. An unknown model is a 400 listing the configured ones. Answers as a stream of server-sent events (below); <code>final</code> names the model that answered (<code>model_profile</code>) and says whether the answer came from the cache (<code>cached</code>)</td></tr>
</tbody>
</table>

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
- **Each answer shows:** "Interpreted as …" for a condensed follow-up and the release used; the answer with clickable `[n]` (highlights the source); the articles and links block; Sources, each with its file path under it; notices; a collapsible "Context sent to the LLM"; retrieval / generation times, model, tokens/s and the trace id; a notice "Answered from the answer cache …" when the same question was answered before under the same conditions (it comes back at once, with the same sources); 👍 / 👎 with an optional comment; after a 👎, a reason can be chosen (Wrong, Incomplete, Should have answered, Should have refused), and it is shown again when the conversation is reopened. `kb calibrate` uses the last two reasons.
- **Behind the scenes** (button under each answer, also in reopened conversations): a side panel with tabs **Retrieval** (every candidate with its rerank and retrieval score and rank, the ones that reached the context highlighted, the filters and the gate), **Context** (the units sent), **Prompt** (the exact messages, the model, tokens and speed, the raw model output), **Checks** (citations kept and removed, the refusal retry, the command check, and a **Check faithfulness** button that runs the judge on demand, 20–60 s) and **Timings** (each stage). Esc or ✕ closes it.
- **Release chip** (above the input): the conversation's release; ✕ clears it. Name a release in a question to set it.
- **Not found** answers are shown in amber, with whether the gate (no relevant documents) or the LLM (documents do not contain the answer) refused.
- **Health dot** (top): green when SQLite, Qdrant, Ollama and the models are all fine; hover for details.

**Chat turn** (`POST /api/chat`), in order:

1. **Release:** a release in the question (`R2025x`) becomes the conversation's filter and stays for later questions; "any release" / "all releases" clears it; a question naming two releases (a comparison) is not filtered.
2. **Follow-ups:** a question that depends on earlier turns ("and on Oracle?", "how do I change it?") is rewritten by the local LLM into a standalone question from the last 6 messages. Self-contained questions skip this. Condensing always runs locally (the conversation may contain text not cleared for external LLMs); a bad rewrite falls back to the original question.
3. **Answer:** the same pipeline as `kb ask` (retrieval with the user's groups and the release, gate, LLM, citations, article numbers / URLs).
4. **Stored:** the question (with its standalone version) and the answer (with its trace id) in `messages`; the trace keeps the original and the standalone question and a `condense` stage.

Server-sent events, in order: `session` (session id, sticky release) → `status` (`queued` while another turn is running, `condensing`, then `searching` with the standalone question and release; for a comparison also `comparing` while the question is split, then `searching_side` with `side`, `index` and `total` before each side's search (`reading` with the `sections` added by the comparison read step, only when the answer model has `compare_read: true`); `retrying` with `units` when the model found no answer and is asked again with fewer sources, followed by a second `context` event and more `token` events) → `context` (the numbered sources sent to the LLM, each with its file `path` as set by `KB_SOURCE_PATH`; `same_text` lists near-identical copies in other documents that were not sent again) → `token` … (the raw answer as it streams) → `final` (the **cleaned** answer, `route` and the compared `sides`, sources (each with its file `path`, full or relative per `KB_SOURCE_PATH`, and its `same_text` copies), article references, notices, gate verdict, provider, timings, trace id, and `cached`: true when the answer came from the answer cache), or `error` instead of `final`. A UI should replace the streamed text with `final.answer`. Only one turn runs at a time (shared models and GPU); a request that has to wait gets `status: queued` first and starts when the running turn ends. If the browser disconnects mid-answer, the turn still finishes and is stored: reopening the conversation shows it.

Without the UI: open <http://127.0.0.1:8000/docs>, expand **POST /api/chat**, *Try it out*, and send `{"question": "Which port does the application server use on R2026x?"}` (the docs page shows the events when the answer is complete), then the same with `"session_id"` from the `session` event and `"question": "and on Oracle?"`.
