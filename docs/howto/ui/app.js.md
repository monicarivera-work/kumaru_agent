# `ui/app.js`

> Browser-side controller for Kumaru chat: local conversation state, escape-first markdown rendering, streaming SSE parsing, status polling, and backend hot-swapping.

**Read this when:** you need to change the UI behavior, debug streaming, add a request parameter, alter markdown support, or wire a new DOM control to the API.

---

## What it does

`ui/app.js` turns the static HTML shell into a working chat client. It:

- stores the conversation in `localStorage`;
- renders user text and assistant markdown safely;
- posts OpenAI-compatible chat requests to `/v1/chat/completions`;
- reads streamed Server-Sent Event frames from a `fetch()` response body;
- updates token/latency stats from Kumaru-specific stream metadata;
- polls `/v1/info` for model status and available backends;
- posts `/v1/backend` requests to hot-swap a model/backend;
- wires buttons, chips, keyboard shortcuts, sliders, and copy buttons.

## Why it exists

The server is stateless for conversations. Every chat request carries the whole message history, so the UI must own history, persistence, rendering, and cancellation. Keeping that logic in one plain JavaScript file makes the browser contract easy to audit.

## Mental model

The client owns state; the server owns model execution.

1. `state.messages` is the source of truth for the visible conversation.
2. `persist()` stores the latest 100 turns in `localStorage`.
3. `renderAll()` rebuilds the message list from state.
4. `send()` builds an OpenAI-style `messages` payload, streams the response, then appends the final assistant message.
5. `refreshInfo()` periodically asks the server what is loaded.

Streaming uses `fetch()` plus `ReadableStream`, not `EventSource`, because the request must be a POST with a JSON body.

## Structure walkthrough

| Top-level item | Purpose | Key dependencies |
|---|---|---|
| `$` | Shorthand for `document.getElementById(id)`. | Every id must exist in `ui/index.html`. |
| `state` | Mutable client state: `messages`, `streaming`, `controller`, `info`. | `localStorage["kumaru.messages"]`; `AbortController`. |
| `escapeHtml(s)` | Escapes `& < > " '` before rendering user/model text. | Required for the escape-first markdown security model. |
| `renderMarkdown(src)` | Converts a small markdown subset to HTML after escaping. | Emits tags styled by `.bubble ...` rules in CSS. |
| `messageEl(role, html, meta)` | Creates a `.msg <role>` DOM node with avatar and bubble. | Role affects avatar label: `You`, `!`, or `◈`. |
| `renderAll()` | Rebuilds `#messages` from `state.messages`. | Calls `wireChips()`, `wireCopy()`, and `scrollToBottom()`. |
| `scrollToBottom()` | Scrolls the message container to its bottom. | Requires `#messages`. |
| `persist()` | Saves only the latest 100 messages to `localStorage`. | Prevents unbounded local browser storage growth. |
| `wireChips()` | Adds click handlers to suggestion buttons. | Copies chip text to `#input`, then calls `send()`. |
| `wireCopy()` | Adds copy handlers to code-block buttons. | Uses `navigator.clipboard`; buttons are emitted by `renderMarkdown()`. |
| `send()` | Main chat flow: validate input, append user turn, create payload, stream assistant response, handle errors, persist. | Calls `/v1/chat/completions`; uses slider and system prompt values. |
| `setStreaming(on)` | Toggles streaming state and send/stop/input UI. | Requires `#send-btn`, `#stop-btn`, and `#input`. |
| `updateStats(k)` | Writes time-to-first-token, speed, and token counts. | Called when stream chunk contains `evt.kumaru`. |
| `refreshInfo()` | Polls server status and updates sidebar facts and backend selector. | Calls `/v1/info`; reads `available_backends`, `config`, and `agent`. |
| `init()` | Wires all DOM event handlers, initial render, status polling, and focus. | Runs on `DOMContentLoaded`. |
| `document.addEventListener("DOMContentLoaded", init)` | Starts the UI once DOM nodes exist. | Must remain after defining `init()`. |

### `renderMarkdown()` markdown subset

| Input | Output |
|---|---|
| Fenced code blocks | `<pre><button data-copy>copy</button><code class="lang-...">...</code></pre>` |
| `#`, `##`, `###` headings | `<h1>`, `<h2>`, `<h3>` |
| `> quote` | `<blockquote>` |
| Inline backticks | `<code>` |
| `**strong**` | `<strong>` |
| `*emphasis*` | `<em>` |
| `[text](https://...)` | Safe external link with `target="_blank"` and `rel="noopener"` |
| Consecutive `- ` or `* ` lines | `<ul>` |
| Consecutive `1. ` lines | `<ol>` |
| Other nonblank chunks | `<p>` with single newlines converted to `<br>` |

Raw HTML never passes through because `escapeHtml()` runs before every replacement.

## How to customize it

### Add a request parameter from a new slider

1. Add the control in `ui/index.html`.
2. Bind its visible value in `init()`:

```js
bind("top-k", "val-topk", (v) => v);
```

3. Include it in the request:

```js
body: JSON.stringify({
  messages: payload,
  stream: true,
  temperature: parseFloat($("temperature").value),
  max_tokens: parseInt($("max-tokens").value, 10),
  top_p: parseFloat($("top-p").value),
  top_k: parseInt($("top-k").value, 10),
})
```

4. Verify the FastAPI request schema accepts `top_k`; otherwise the server may reject or ignore it.

### Add a keyboard shortcut

Add a branch to the existing `keydown` listener:

```js
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  if (e.key === "Escape" && state.streaming) state.controller?.abort();
});
```

Keep `Shift+Enter` as the newline path.

### Change the markdown renderer

Add rules after escaping and before paragraph wrapping. Example for horizontal rules:

```js
text = text.replace(/^---$/gm, "<hr>");
```

Then update CSS with a `.bubble hr` rule. Do not add a replacement that allows raw HTML from model output.

### Include API authentication

If `server.api_key` is set, protected endpoints expect `Authorization: Bearer <key>`. Add a local operator-provided token source and include the header in `send()` and the backend swap request:

```js
headers: {
  "Content-Type": "application/json",
  "Authorization": `Bearer ${token}`,
}
```

Do not hard-code secrets in this file.

### Change persistence behavior

`persist()` currently keeps the newest 100 messages:

```js
localStorage.setItem("kumaru.messages", JSON.stringify(state.messages.slice(-100)));
```

Raise or lower that number based on browser storage constraints. This does not change `agent.max_history_tokens`; the server still trims prompt history by token budget.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Cannot read properties of null` in console | An element id expected by `$()` is missing or renamed. | Compare `$("<id>")` calls with `ui/index.html`. |
| Streaming response arrives as one delayed blob | Proxy buffering is enabled or server did not send SSE headers. | Check `X-Accel-Buffering: no` in `routes_chat.py` and proxy config. |
| Model output can inject HTML | Someone moved replacements before `escapeHtml()` or reintroduced raw HTML. | Restore escape-first rendering and avoid raw HTML passthrough. |
| Stop button does not stop GPU work | Abort did not reach the request or server disconnect check is failing. | Verify `AbortController` is used and `_stream_sse()` checks `request.is_disconnected()`. |
| Error messages persist in history | `send()` appends `{ role: "error", ... }` on non-abort failures. | Clear conversation or change error handling if errors should be transient. |
| `/v1/backend` returns “No valid fields supplied” | Payload keys are not in the server allow-list. | Use one of `name`, `model`, `device`, `dtype`, `load_in_4bit`, `adapter_path`, `base_url`, or `model_path`. |
| Backend selector stays empty | `/v1/info` failed or returned no `available_backends`. | Inspect server logs and test `GET /v1/info`. |
| Copy button fails | Browser blocked `navigator.clipboard` or page is not in a secure context. | Use localhost/HTTPS and check browser permissions. |
| Old conversations reappear | `localStorage["kumaru.messages"]` still contains history. | Use Clear conversation or clear that key in dev tools. |

## Related files

- `ui/index.html` — DOM ids and static markup used by this file.
- `ui/styles.css` — styles generated `.msg`, `.bubble`, `.copy-btn`, `.cursor`, and `.hidden` classes.
- `src/kumaru/server/routes_chat.py` — implements the endpoints and SSE frame shape consumed here.
- `src/kumaru/core/types.py` — defines shared generation and message concepts.
- `src/kumaru/core/config.py` — defines backend and generation config surfaced by `/v1/info`.

