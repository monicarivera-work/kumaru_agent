# `ui/index.html`

> Static HTML shell for the Kumaru web chat UI, including the sidebar controls and message composer that `ui/app.js` wires to the API.

**Read this when:** you need to add a visible control, rename an element, debug a missing DOM binding, or change the initial page copy.

---

## What it does

`ui/index.html` defines the complete browser UI. The server serves it as a static file; there is no build step and no template engine. It loads `/static/styles.css` and `/static/app.js`, then `app.js` binds behavior to the element ids declared here.

The page has two regions:

- `#sidebar`: status, runtime facts, generation sliders, system prompt, backend swap controls, and footer actions.
- `#main`: conversation history, empty-state suggestion chips, input textarea, send/stop buttons, and hint text.

## Why it exists

Kumaru is local-first and intentionally simple to operate. Keeping the UI in one HTML file means operators can audit every browser-facing control and trace each one to a JavaScript binding or API call without compiling assets.

## Mental model

Treat this file as the DOM contract for `ui/app.js`.

1. HTML declares ids and static copy.
2. CSS controls presentation.
3. JavaScript reads and mutates the elements by id.
4. The server exposes `/v1/info`, `/v1/chat/completions`, and `/v1/backend`.

If you change an id here, update every `$("<id>")` lookup in `ui/app.js`.

## Structure walkthrough

| Section / id | Purpose | JavaScript dependency |
|---|---|---|
| `<!DOCTYPE html>`, `<html lang="en">`, viewport meta | Standards-mode, responsive HTML document. | None. |
| `<link rel="stylesheet" href="/static/styles.css">` | Loads the hand-written stylesheet. | None. |
| Inline SVG favicon | Browser tab icon; no file request needed. | None. |
| `#app` | Grid container for sidebar and main chat area. | None directly; styled by CSS. |
| `#sidebar` | Left column for model status and controls. | Contains most ids used by `refreshInfo()` and `init()`. |
| `.brand`, `.mark`, `#version` | Product mark and version/status line. | `refreshInfo()` writes `v${info.version}` into `#version`. |
| Status panel: `#status-dot`, `#status-text` | Shows loading, ready, or error state. | `refreshInfo()` toggles dot classes and text. |
| `#model-facts` | Definition list of loaded model facts. | Container only. |
| `#fact-model` | Effective model id. | Filled from `info.agent.backend.model_id` or `info.config.backend.model`. |
| `#fact-backend` | Effective backend name. | Filled from backend runtime/config info. |
| `#fact-device` | Runtime device such as `cuda` or `cpu`. | Filled from backend runtime/config info. |
| `#fact-dtype` | Runtime dtype such as `bfloat16`. | Filled from backend runtime/config info. |
| `#fact-context` | Loaded context window length. | Filled from backend runtime info. |
| Last response panel: `#stat-ttft` | Time to first token for the most recent stream. | Updated by `updateStats(k)`. |
| `#stat-tps` | Tokens per second for the most recent stream. | Updated by `updateStats(k)`. |
| `#stat-tokens` | Prompt-to-completion token count. | Updated by `updateStats(k)`. |
| Decoding panel: `#temperature`, `#val-temp` | Sampling temperature slider and displayed value. | Bound in `init()`; sent as `temperature`. |
| `#max-tokens`, `#val-max` | Reply token cap slider and displayed value. | Bound in `init()`; sent as `max_tokens`. |
| `#top-p`, `#val-topp` | Nucleus sampling slider and displayed value. | Bound in `init()`; sent as `top_p`. |
| `#system-prompt` | Optional client-side system prompt prepended to requests. | `send()` adds it as a `system` message when non-empty. |
| Model panel: `#swap-model` | Text input for a new model id. | Sent to `POST /v1/backend` as `model`. |
| `#swap-backend` | Backend selector populated at runtime. | `refreshInfo()` fills it from `available_backends`; swap sends `name`. |
| `#swap-btn` | Hot-swap trigger. | Click handler posts to `/v1/backend`. |
| `#clear-btn` | Clears local conversation history. | Click handler resets `state.messages` and `localStorage`. |
| `/docs` link | Opens FastAPI-generated API docs. | None. |
| `#main` | Right column for chat history and composer. | Styled by CSS; contains message and composer ids. |
| `#messages` | Message list container. | `renderAll()`, `send()`, and `scrollToBottom()` read/write it. |
| `.empty-state` | Initial no-history display. | Recreated by `renderAll()` when history is empty. |
| `.chip` buttons | Suggested prompts. | `wireChips()` copies text into `#input` and calls `send()`. |
| `#composer` | Bottom input region. | Styled only. |
| `#input` | User message textarea. | `init()` auto-resizes, handles Enter, disables while streaming. |
| `#send-btn` | Sends the current input. | Click handler calls `send()`; hidden during streaming. |
| `#stop-btn` | Cancels an in-flight stream. | Click handler aborts `state.controller`; hidden when idle. |
| `#hint` | Small status hint under composer. | `refreshInfo()` writes loading/error/ready hints. |
| `<script src="/static/app.js">` | Loads client behavior after DOM markup. | Must remain after elements or wait for `DOMContentLoaded`. |

## How to customize it

### Change the initial suggestions

Edit the three `.chip` buttons in the empty state:

```html
<button class="chip">Summarize this stack trace</button>
<button class="chip">Draft a concise incident update</button>
<button class="chip">Explain this YAML config</button>
```

Also update the duplicate empty-state markup in `renderAll()` in `ui/app.js`; otherwise suggestions revert after clearing the conversation.

### Add a sidebar control that affects chat requests

1. Add an element in the Decoding panel:

```html
<label class="slider">
  <span>top-k <b id="val-topk">50</b></span>
  <input type="range" id="top-k" min="0" max="200" step="1" value="50" />
</label>
```

2. In `ui/app.js`, bind the display in `init()`:

```js
bind("top-k", "val-topk", (v) => v);
```

3. Add `top_k: parseInt($("top-k").value, 10)` to the `/v1/chat/completions` request body.

4. Confirm the API schema accepts the new request field before depending on it.

### Add another backend-swap option

Do not hard-code `<option>` elements here. `refreshInfo()` fills `#swap-backend` from `/v1/info.available_backends`, which comes from the backend registry. Register the backend server-side first.

### Change the API docs link

The current link assumes FastAPI docs live at `/docs`. If you serve the UI behind a path prefix, change:

```html
<a href="/docs" target="_blank" rel="noopener">API docs ↗</a>
```

to the deployed docs path.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Browser console shows `Cannot read properties of null` | An id in this file no longer matches `ui/app.js`. | Search `ui/app.js` for `$("...")` and restore or update the id. |
| Sliders display values but requests ignore them | The input exists, but `send()` does not include the value in the JSON body. | Add the field to the `/v1/chat/completions` body and verify the server schema accepts it. |
| Suggestions disappear or come back after clear | Only one copy of the empty-state markup was changed. | Update both `index.html` and `renderAll()` in `ui/app.js`. |
| Load Model button has no backend choices | `/v1/info` failed or returned no `available_backends`. | Check server logs and `GET /v1/info`. |
| Stop button never appears | `setStreaming(true)` is not running or ids changed. | Verify `#send-btn`, `#stop-btn`, and `send()` error handling. |
| Sidebar hidden on a tablet-sized viewport | CSS media query hides `#sidebar` below 860px. | Adjust `@media (max-width: 860px)` in `ui/styles.css`. |

## Related files

- `ui/app.js` — binds behavior to the ids in this file.
- `ui/styles.css` — styles the sidebar, message list, markdown output, and composer.
- `src/kumaru/server/routes_chat.py` — implements `/v1/info`, `/v1/chat/completions`, and `/v1/backend`.
- `src/kumaru/core/config.py` — defines backend, generation, agent, server, and telemetry config fields.

