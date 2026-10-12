# `ui\app.js`

> The whole browser client in one 604-line classic script. It keeps the conversation in `localStorage`, reads the SSE stream by hand with thoughts kept apart from the answer, and drives the Think pill and sampling controls. It also shows per-reply stats, polls server status and hot-swaps the backend. No framework, no build, no dependencies.

**Read this when:**
- you change what the UI sends (a new control, a different default) or how it reads the stream;
- thoughts, stats or errors render wrong, or the sidebar sits at "connecting…";
- you add a markdown feature, a keyboard shortcut, API-key auth, or persistence.

---

## What it does

`ui\app.js` turns the static shell in `ui\index.html` into a chat client. The server serves it at `/static/app.js`. It is a classic (non-module) script loaded at the end of `<body>`, and it starts on `DOMContentLoaded` (line 604). It:

- keeps the conversation in `state.messages` and saves the last 100 messages to `localStorage["kumaru.messages"]`;
- sends every turn to `POST /v1/chat/completions` with `stream: true` and parses the Server-Sent Events itself;
- shows thoughts (`delta.reasoning_content`) in a collapsible **Thinking…** panel above the answer. The answer (`delta.content`) renders as escape-first markdown;
- turns the Think pill, the thinking-budget slider and **Use the model's tuned settings** into request fields;
- copies the stats object `kumaru` from the final frame into the **Last response** panel and the reply's footer line;
- polls `GET /v1/info` every 4 s for status, model facts and the profile's sampling defaults;
- hot-swaps the backend through `POST /v1/backend`.

It only ever talks to its own origin, through absolute paths (`/v1/…`).

## Why it exists

The Python server serves the UI straight from the repo's `ui\` folder, and the UI must work offline. So it is zero-build vanilla JS: no npm, no bundler, nothing to download; edit the file and reload. The header comment (lines 3–24) states four rules; it says "Three things" but lists four:

1. **The client owns the conversation.** Every request carries the full history, as with the OpenAI API. The server stores nothing, so replicas need no session affinity. Persistence lives in `localStorage`, in this file.
2. **Streaming uses `fetch` + `ReadableStream`, not `EventSource`.** `EventSource` can only GET, but a chat turn is a POST with a JSON body.
3. **Markdown is rendered escape-first.** Model output is untrusted input. All HTML is escaped before any markup is generated, so a reply cannot inject elements or scripts.
4. **Thoughts are shown, never sent back.** They are displayed and saved locally, but the outbound history carries answers only. Models are trained on histories whose earlier turns hold answers, not thoughts. Qwen3-style templates discard old thoughts anyway, and resending them burns context.

The v0.2 decisions, and the problem each one solves:

| Decision | Where | Why |
|---|---|---|
| **Use the model's tuned settings** is on by default and then omits `temperature`, `top_p` and `max_tokens` | `requestSettings()` 401 | The server profile has one preset for normal replies (`generation`) and one for thinking replies (`thinking_mode`). A UI that always sent its slider values would override both. |
| Think is a pill on the message box, not a sidebar setting | `index.html:121`, `init()` 568 | Thinking trades speed for quality on hard problems ("slower, meant for hard maths, code and logic", `index.html:50-51`), so it is chosen per question. |
| Sliders step through hand-picked stops (`MAX_TOKEN_STEPS`, `BUDGET_STEPS`) | lines 30, 33 | Token counts span two orders of magnitude. A linear range would be unusable at the low end. |
| Repaint at most once per animation frame | `paint()` 264 | Otherwise a fast GPU triggers hundreds of layout passes per second, and the UI stutters even though the model keeps up. |
| Thoughts are appended to one text node with `appendData` | 274 | Each frame costs the same however long the thought grows, and a text node never parses HTML. |
| The live reply bubble is built once and patched in place | 234–248 | Rebuilding it each frame would reset the user's open/closed choice on the thinking panel. |
| Auto-scroll only when the reader is near the bottom: 120 px for the page, 40 px inside the thought | `isNearBottom()` 169, 273 | A reader who scrolled up to reread something is not yanked back down. |
| `scrollTo({behavior: "instant"})` | `scrollToBottom()` 160 | A smooth scroll restarted 60 times a second never catches up, and it makes `isNearBottom()` lie mid-animation. |
| Thinking time prefers the server's `thinking_ms` | `finished()` 388 | It has no network jitter. The client clock is only the fallback. |
| If saving fails, save again without `reasoning` | `persist()` 173 | Long thoughts can exceed the ~5 MB `localStorage` quota, and answers matter more than thoughts. |

## Mental model

```text
 controls ──► syncControls() ◄── refreshInfo() every 4 s ◄── GET /v1/info
 (pill, budget,     │ (labels, disabled state,               (status, facts, profile
  tuned box,        │  aria-pressed)                          generation/thinking_mode)
  sliders)          ▼
                 state { messages, think, thinkTouched, controlsReady, info, streaming, controller }
                    │
 send() ──► POST /v1/chat/completions  {messages: answers only, stream: true, ...requestSettings()}
                    │
   SSE:  role frame ─► reasoning_content … ─► content … ─► finish frame {kumaru: stats} ─► [DONE]
            (ignored)   .thinking-body        .answer        updateStats() + meta line
                    │
 finally ─► state.messages.push(finished(…)) ─► persist() ─► renderAll()
```

Hold five ideas:

1. **`state` (lines 35–43) is the only model.** `state.messages` is the conversation. `renderAll()` rebuilds `#messages` from it after every change. During a stream, a single live bubble is patched in place instead; `renderAll()` replaces it when the stream ends.
2. **Stored messages have three shapes.**
   - User: `{role: "user", content}`.
   - Assistant: `{role: "assistant", content, meta, reasoning?, thinkingMs?, note?}`.
   - Error: `{role: "error", content}`, markdown that begins `**Request failed:**`.
   - Only user and assistant messages with non-empty `content` are sent back, reduced to `{role, content}`.
3. **The server decides sampling unless told otherwise.** With tuned settings on, the request carries only `thinking` and perhaps `thinking_budget`. The server resolves the rest from its profile.
4. **`syncControls()` is the single place where control state becomes DOM state.** Every control handler calls it, and so does every poll.
5. **Status is polled, not pushed.** Status, facts, the backend list and the initial Think state all come from `/v1/info`.

## Structure walkthrough

| Line | Top-level item | Purpose | Key dependencies |
|---|---|---|---|
| 27 | `$(id)` | `document.getElementById` shorthand | — |
| 30 | `MAX_TOKEN_STEPS` | 8 stops: 256, 512 … 32768. The `#max-tokens` value is an index into it. | `#max-tokens` (`max="7"`) |
| 33 | `BUDGET_STEPS` | 9 stops: 256 … 16384, 24576, then `null` (unlimited). Kept below the 32k reply cap so an answer still fits after the budget is spent. | `#thinking-budget` (`max="8"`) |
| 35 | `state` | `messages` (parsed from `localStorage`), `streaming`, `controller`, `info`, `think`, `thinkTouched`, `controlsReady` | `localStorage` |
| 47 | `escapeHtml(s)` | Escapes `& < > " '` | — |
| 53 | `renderMarkdown(src)` | Escape-first markdown subset (see below) | `escapeHtml` |
| 102 | `formatSeconds(ms)` | `4.2s` below 10 s, `37s` below 60 s, else `2m 5s` | — |
| 110 | `thinkingHtml(text, label)` | Closed `<details class="thinking">`. Label and text are escaped and never parsed. | `escapeHtml` |
| 115 | `assistantHtml(m)` | Thinking panel ("Thought for X", or "Thoughts" when no time is known), then the answer. An empty answer shows `<p class="note">` with `m.note` or "(empty response)". | `thinkingHtml`, `renderMarkdown` |
| 125 | `messageEl(role, html, meta)` | `.msg.<role>` with an avatar (`You`, `◈`, `!`) and an optional `.msg-meta` footer. The footer is inserted as raw HTML; it is built only from numbers and fixed text. | — |
| 135 | `renderAll()` | Rebuilds `#messages`. With no messages it writes the empty state and three suggestion chips. User text is escaped with `\n` → `<br>` (no markdown); error rows are markdown. | `wireChips`, `wireCopy`, `scrollToBottom` |
| 160 | `scrollToBottom()` | Scrolls instantly to the end | `#messages` |
| 169 | `isNearBottom(el, slack)` | Auto-scroll test; defaults to `#messages` and 120 px | — |
| 173 | `persist()` | Saves the last 100 messages. On failure it retries without `reasoning`, then gives up silently. | `localStorage` |
| 188 | `wireChips()` | Chip click → copy the chip text into `#input` → `send()` | `.chip` |
| 197 | `wireCopy()` | Code-block copy buttons; the label reads "copied" for 1.2 s | `navigator.clipboard` |
| 209 | `send()` | One turn: payload, live bubble, fetch, SSE loop, save | nearly everything |
| 388 | `finished(...)` | Stored form of a reply. `thinkingMs` = the server's figure ?? the client clock. `note` is kept only when the answer is empty. | — |
| 401 | `requestSettings()` | Control state → request fields (see the table below) | `#thinking-budget`, `#auto-sampling`, sliders |
| 415 | `setStreaming(on)` | Swaps Send and Stop, and disables `#input` | `.hidden` |
| 422 | `updateStats(k)` | Final-frame stats → **Last response** panel | `#stat-*` |
| 431 | `nearestIndex(steps, value)` | Nearest stop on a log scale, skipping `null` | — |
| 442 | `profileDefaults()` | `config.generation`, overlaid with the non-null `config.thinking_mode` fields when Think is on | `state.info` |
| 454 | `syncControls()` | Labels, disabled states, `.is-disabled` and `aria-pressed` | every control |
| 486 | `refreshInfo()` | Status dot and text, hint, model facts, version, backend list, initial Think state | `GET /v1/info` |
| 544 | `init()` | Wires the events, renders, polls, starts the 4 s interval (600) and focuses `#input` (601) | DOM |
| 604 | `DOMContentLoaded` listener | Entry point | — |

### `send()` step by step

1. **Guard (212).** Return if the trimmed input is empty or a stream is running. Otherwise clear the input, push the user message, `renderAll()`, `persist()`.
2. **Payload (222–229).** The trimmed `#system-prompt`, if any, goes first as a `system` message; the server merges it with its own prompt. Then comes every user and assistant message with non-empty `content`, as `{role, content}`. Error rows, notes, thoughts and empty replies are never sent.
3. **Live bubble (236–249).** `<details class="thinking live hidden" open>` followed by `<div class="answer">` holding a blinking cursor. It is appended to `#messages` and scrolled into view.
4. **Request (292–297).** `fetch` with a `signal` from a fresh `AbortController`. The only header is `Content-Type: application/json`. The body is `{messages, stream: true, ...requestSettings()}`.
5. **HTTP errors (299–302).** Throw the body's `error.message`, or `HTTP <status>` when there is none.
6. **Read loop (308–356).**
   - `TextDecoder.decode(value, {stream: true})` keeps a multi-byte character that is split across network chunks intact.
   - The buffer is split on `\n\n` and the trailing fragment is kept for the next read.
   - Each frame contributes its first `data:` line. `[DONE]` and unparsable JSON are skipped.
   - Fields are routed as in the table below.
7. **End of stream (358–367).** `finish_reason === "length"` appends " · cut off at max tokens" to the meta line. If no answer arrived, it also sets a note: "No answer: the max-tokens cap ran out while the model was still thinking. Raise max tokens or set a thinking budget." (or "No answer: the max-tokens cap ran out." without thoughts). Then the reply is pushed through `finished()`.
8. **Abort (369–374).** Stop keeps whatever arrived: meta `stopped`, note "Stopped before answering.". If nothing arrived, nothing is saved.
9. **Any other error (376).** Pushes an error row. **The partial reply is discarded.**
10. **`finally` (378–383).** `setStreaming(false)`, `persist()`, `renderAll()`. Focus is not given back to `#input`.

| Frame content | Effect in `send()` |
|---|---|
| `{"error": {message, type, hint}}` (in-band, after headers were sent) | throw → error row; `hint` is dropped |
| `choices[0].delta.role` only (first frame) | nothing |
| `choices[0].delta.reasoning_content` | The first one starts the client thinking clock. Text is appended to `reasoning`. |
| `choices[0].delta.content` | The first one after thoughts stops the clock, closes the panel (`details.open = false`) and removes `live`. Text is appended to `acc`. |
| `choices[0].finish_reason` | remembered for the cut-off note |
| `kumaru` (finish frame only) | `updateStats()`, `thinkingMs`, meta line |

The parser relies on Kumaru's own framing: `\n\n` separators and exactly one single-line `data:` per frame (`frame()` inside `_stream_sse`, `src\kumaru\server\routes_chat.py`). That is not SSE in general. Use a full SSE parser before pointing this code at another server.

### The thinking panel

| Moment | DOM | Summary label |
|---|---|---|
| Request sent, nothing back | `.thinking.live.hidden[open]` is invisible; the cursor blinks in `.answer` | — |
| First thought | `hidden` removed; text streams into `.thinking-body`; the answer cursor is removed | `Thinking… Ns`: whole seconds on the client clock, redrawn on each paint |
| First answer token | `open = false`, `live` removed (the label stops pulsing) | `Thought for X` on the client clock |
| Stream ends | `renderAll()` swaps in `thinkingHtml()`, closed | `Thought for X` from the server's `thinking_ms`, else "Thoughts" |

- While streaming, the user can open or close the panel and the choice sticks. The bubble is never rebuilt; only the first answer token closes it, once.
- `.thinking-body` scrolls inside a 280 px box (`styles.css`). It auto-follows only while its scroll position is within 40 px of the bottom.
- Thoughts are never parsed as markdown. The live view is a text node and the stored view is escaped. Both are shown with `white-space: pre-wrap`.
- The panel appears whenever reasoning frames arrive, including from a model that thinks while Think is off.

### Think, budget and sampling → request fields

`requestSettings()` (401–413):

| Control state | Fields sent |
|---|---|
| Think off | `thinking: false` |
| Think on, budget slider at **unlimited** (the default, index 8) | `thinking: true` |
| Think on, budget at a stop | `thinking: true, thinking_budget: <256 … 24576>` |
| Tuned settings ticked (default) | no `temperature`, `top_p` or `max_tokens` |
| Tuned settings unticked | `temperature` and `top_p` from the sliders, `max_tokens: MAX_TOKEN_STEPS[index]` |

Never sent: `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, `stop`, `seed`, `model`, `chat_template_kwargs`, `reasoning_effort`. The budget slider is disabled while Think is off, and its value is ignored then.

How the server combines the request with its profile, in `ChatAgent.resolve_params()` (`src\kumaru\agent\chat.py:81`):

1. Start from `generation`.
2. If thinking resolves to true, overlay `thinking_mode`.
3. Overlay every non-null request field.

After the history is trimmed, `ChatAgent._fit()` (`chat.py:100-113`) lowers `max_tokens` to the room the prompt leaves in the context window. A large slider value therefore cannot make a strict server reject the request.

`profileDefaults()` mirrors steps 1–3 so that the greyed-out sliders show what will actually be used. Three consequences follow:

- **`thinking` is always a boolean.** The UI can never defer to a profile that sets `generation.thinking: null` ("model default"). The pill starts false, and the first `/v1/info` that carries `config` sets it to `generation.thinking === true` unless the user has already clicked it (`thinkTouched`).
- **"unlimited" means "no budget field".** A profile `thinking_mode.thinking_budget` or `generation.thinking_budget` still applies. The thinking is truly unlimited only when the profile's budget is null.
- **Labels show exact profile values, but the knobs can only park on what the slider can represent.**
  - `max_tokens` parks on the log-nearest stop. A profile value of 3000 shows "3,000" while the knob sits at 4096, so unticking the box sends 4096.
  - A profile temperature above 1.5, or a `top_p` below 0.1, is clamped by the range input.

What each backend does with the two thinking fields:

| Backend | `thinking` | `thinking_budget` |
|---|---|---|
| `llamacpp_server` | `chat_template_kwargs.enable_thinking` (inherited) | `reasoning_budget_tokens` plus `reasoning_budget_message`, unless `thinking` is `false` (`llamacpp_server.py:505-507`) |
| `openai_compat` | `chat_template_kwargs.enable_thinking`; not sent with `backend.extra.strict_openai: true` (`openai_compat.py:138,165-168`) | not sent |
| `transformers` | `enable_thinking` chat-template variable (`transformers_backend.py:285`) | ignored (`:353`) |
| `llamacpp` (in-process) | not passed. Inline `<think>` blocks are still split out (`llamacpp_backend.py:128`), so a model that thinks by default thinks with Think off. | ignored |
| `echo` | a canned thought, only when `thinking` is true (`echo.py:73`) | caps the thought at N 8-character pieces (`echo.py:81-82`) |

A model without a thinking mode ignores `enable_thinking` and answers directly.

### Stats

| Panel row | Element | From the finish frame's `kumaru` | Not updated when |
|---|---|---|---|
| first token | `#stat-ttft` | `time_to_first_token_ms` → `N ms` | the value is null (nothing was generated) |
| thinking | `#stat-think` | `thinking_ms` → `formatSeconds()`, or `off` when null | always updated |
| speed | `#stat-tps` | `tokens_per_second` → `N tok/s` | the value is null |
| tokens | `#stat-tokens` | `prompt_tokens → completion_tokens` | `completion_tokens` is null |

The footer line under each reply (`.msg-meta`) has three parts:

- always: `<tps> tok/s · first token <ms>ms · <n> tokens`;
- ` · thought <time>` when the model thought;
- ` · cut off at max tokens` when `finish_reason` is `length`.

A stopped reply's footer is just `stopped`.

What the numbers mean (`_stream_sse`, `src\kumaru\server\routes_chat.py:113-210`):

- **The clock starts before the request waits for a concurrency slot** (`server.max_concurrent_requests`). TTFT and tok/s therefore include queueing and prompt prefill.
- **First token** is the first thought or answer token, whichever comes first.
- **Thinking** runs from the first thought to the first answer token, or to the finish if no answer came.
- **tok/s** is `completion_tokens ÷ latency`: a whole-request average, not decode speed. `completion_tokens` is whatever the backend's usage reports; echo counts thought frames too (`test_streamed_thoughts_arrive_before_the_answer`).
- **Stop and errors produce no finish frame**, so the panel keeps the previous reply's numbers.

### Errors

| Server outcome | What the user sees |
|---|---|
| 400 `RequestError` (e.g. empty `messages`) | error row `**Request failed:** <message>` |
| 401 `AuthError` (only when `server.api_key` is set; this file sends no key) | `Request failed: Invalid or missing API key.` |
| 503 `BackendError` from `ensure_ready()` | `Backend failed to load: …`, or `Backend is still loading.` after waiting 600 s |
| 422 FastAPI validation (a value out of range) | `Request failed: HTTP 422`. The body has `detail`, not `error.message`. |
| In-band `error` frame after streaming began (e.g. `ContextLengthExceededError` under `agent.memory_policy: none`, an upstream failure) | its message as an error row; the partial reply is discarded |
| Network failure | the browser's text, e.g. `Failed to fetch` in Chromium |
| Model swap failure | `alert("Could not load model: <message>")` |

The server also sends a `hint` with Kumaru errors (for example "Retry in a few seconds."). This file drops it; see "Show the server's hint" below.

### Persistence

| Key | Store | Value |
|---|---|---|
| `kumaru.messages` | `localStorage` | JSON array of the last 100 stored messages (shapes above), saved after every turn |

That is the only key. The Think state, sliders, tuned-settings box, budget and system prompt reset on reload. Think then re-reads the profile default.

- `localStorage` is per origin: `http://127.0.0.1:8000`, `http://localhost:8000` and another port each have their own history.
- The load at line 36 is an unguarded `JSON.parse`. A corrupt value throws while the script loads, so `init` is never registered and the page stays static.

### Keyboard and accessibility

| Input | Effect |
|---|---|
| Enter in `#input` | send (553). There is no `isComposing` check, so Enter that confirms an IME candidate also sends. |
| Shift+Enter | newline; the box grows to 200 px, then scrolls |
| Escape | nothing (there is no handler) |
| Tab to the pill, Space/Enter | toggles Think (it is a real `<button type="button">`) |
| Tab to a thinking summary, Enter | opens or closes it (native `<details>`) |

- `#input` is focused on load (601). It is disabled while streaming and not refocused after, so the keyboard user must Tab back.
- The pill's `aria-pressed` is kept in sync by `syncControls()` (481).
- Nothing is announced to screen readers. Neither `#messages` nor `#status-text` is an `aria-live` region.
- The range inputs expose stop indices (0–7, 0–8), not token counts; there is no `aria-valuetext`.
- Copy buttons are visible only on hover (`styles.css:177-180`), so a keyboard user can Tab onto an invisible button.

### `renderMarkdown()` markdown subset

Rules run on already-escaped text, in this order: fences, headings, quotes, inline code, bold, italics, links, lists, paragraphs. Verified behaviour:

| Input | Output | Notes |
|---|---|---|
| ```` ```lang ```` … ```` ``` ```` | `<pre>` with a copy button and `<code class="lang-lang">` | Pulled out first, so no other rule touches it. An unclosed fence (mid-stream) shows as literal text until it closes. |
| `# `, `## `, `### ` at line start | `<h1>`–`<h3>` | |
| `> ` at line start | `<blockquote>` | one blockquote per line |
| `` `code` `` | `<code>` | not protected from later rules: `` `a * b * c` `` gets an `<em>` inside |
| `**bold**` | `<strong>` | within one line |
| `*italic*` after line start, whitespace or `(` | `<em>` | `_italic_` is not supported |
| `[text](https://…)` | `<a target="_blank" rel="noopener">` | `http:`/`https:` only; `[x](javascript:…)` stays literal |
| lines starting `- ` or `* ` | `<ul>` | Flat only; an indented item becomes plain text. Use `-`: a `*` item that contains `*italic*` is mis-parsed. |
| lines starting `1. ` | `<ol>` | flat only |
| blank line / single newline | new `<p>` / `<br>` | |

Not supported: tables (styled in `styles.css` but never generated), `---` rules, nested lists, images, `_italic_`. Raw HTML is escaped and shown literally, by design.

## How to customize it

### Develop against the echo backend

```powershell
.\.venv\Scripts\kumaru.exe serve -c test --port 8790
```

Open `http://127.0.0.1:8790/`. The echo backend needs no model and no GPU, and its reply reports the parameters it received:

```text
[echo:9b96a1fe] You said: hi there
Context: 2 message(s), ~219 prompt tokens, max_tokens=256, temperature=0.0.
```

That is `configs\test.yaml` with tuned settings on; untick the box and the slider values appear instead. With Think on, a canned thought streams through the reasoning channel. It is about 21 pieces, so even the smallest budget stop (256) never truncates it. Avoid **Load model** here: it closes the running backend first.

After editing, hard-reload with Ctrl+F5. Static files carry `ETag` and `Last-Modified` but no `Cache-Control`, so a normal reload may reuse the old copy.

### Add a request parameter (worked example: `top_k`)

1. **`ui\index.html`**, in the Decoding panel:

   ```html
   <label class="slider">
     <span>top-k <b id="val-topk">50</b></span>
     <input type="range" id="top-k" min="0" max="100" step="1" value="50" />
     <small>Sample from the k most likely tokens; 0 = off.</small>
   </label>
   ```

2. **`requestSettings()`**, inside the `!$("auto-sampling").checked` branch:

   ```js
   s.top_k = parseInt($("top-k").value, 10);
   ```

3. **`syncControls()`**:
   - add `"top-k"` to the disable loop's array;
   - inside `if (eff)`: `$("top-k").value = eff.top_k;`;
   - with the other labels: `$("val-topk").textContent = eff ? eff.top_k : $("top-k").value;`.
4. **`init()`**: add `"top-k"` to the `input` listener list.

`top_k` already exists in the request schema (`src\kumaru\server\schemas.py:62`). A new field must be added there too. The model uses `extra="ignore"`, so a misspelled field returns 200 and silently changes nothing. Prove the value arrives with a spy in `tests\test_server.py`:

```python
from kumaru.backends.echo import EchoBackend

def test_top_k_reaches_the_backend(test_client, monkeypatch):
    seen = []
    original = EchoBackend.stream

    def spy(self, messages, params):
        seen.append(params.top_k)
        return original(self, messages, params)

    monkeypatch.setattr(EchoBackend, "stream", spy)
    test_client.post("/v1/chat/completions", json={**chat_payload(), "top_k": 7})
    assert seen == [7]
```

`EchoBackend` defines only `stream()`; the base `generate()` consumes it, so the spy also sees non-streaming calls.

### Keyboard: Escape to stop, IME-safe Enter, focus after a reply

```js
// init(): #input is disabled while streaming, so listen on the document.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.streaming) state.controller?.abort();
});

// init(), the #input keydown handler: never send while an IME composition is open.
if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }

// send(), at the end of the finally block: give the keyboard back.
$("input").focus();
```

### Add a markdown rule (worked example: horizontal rules)

Every rule sees escaped text: match `&gt;`, `&lt;`, `&amp;`, `&quot;`, `&#39;`, never the raw characters, and never un-escape.

1. In the chain after the heading rules, add:

   ```js
   .replace(/^(?:-{3,}|\*{3,})$/gm, "<hr>")
   ```

2. Add `hr` to the paragraph-skip test: `/^<(h[1-3]|ul|ol|pre|blockquote|table|hr)/`.
3. In `ui\styles.css`, add:

   ```css
   .bubble hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
   ```

`"above\n\n---\n\nbelow"` then renders as `<p>above</p>`, `<hr>`, `<p>below</p>`. A rule needs blank lines around it; otherwise it lands inside a paragraph.

### Send an API key

With `server.api_key` set, chat and swap require `Authorization: Bearer <api_key>`. The UI files and `/v1/info` stay public. Never hard-code the key in this file: anyone who can load the page can read it. Keep it in tab-scoped storage instead:

```js
function authHeaders() {
  const key = sessionStorage.getItem("kumaru.apiKey");
  return key ? { Authorization: "Bearer " + key } : {};
}
// In send() and in the swap handler:
//   headers: { "Content-Type": "application/json", ...authHeaders() },
```

Set it once per tab from the DevTools console: `sessionStorage.setItem("kumaru.apiKey", "<api_key>")`.

### Remember the Think choice across reloads

```js
// init(), before the first syncControls():
const saved = localStorage.getItem("kumaru.think");
if (saved !== null) { state.think = saved === "true"; state.thinkTouched = true; }

// The think-btn click handler, after toggling:
localStorage.setItem("kumaru.think", String(state.think));
```

Setting `thinkTouched` stops `refreshInfo()` from overwriting the saved choice with the profile default.

### Show the server's hint with errors

```js
// The !res.ok branch:
const e = err?.error;
throw new Error([e?.message, e?.hint].filter(Boolean).join(" — ") || `HTTP ${res.status}`);

// The in-band branch:
if (evt.error) throw new Error([evt.error.message, evt.error.hint].filter(Boolean).join(" — "));
```

### Change persistence

| Change | Where |
|---|---|
| History length | `slice(-100)` in `persist()` |
| Never store thoughts | serialize `recent.map(({ reasoning, ...m }) => m)` on the first attempt |
| Rename the key | lines 36, 176 and 181 |
| Survive corrupt storage | wrap line 36's `JSON.parse` in `try`/`catch` and fall back to `[]` |

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Page renders unstyled | `styles.css` served as `text/plain`. Python takes MIME types from the Windows registry, where some installers map `.css`/`.js` to `text/plain`, and browsers refuse such a stylesheet. `app.js` would still run: it is a classic script and the server sends no `nosniff`. A module script or `nosniff` would block it too. | Pinned in `src\kumaru\server\app.py:46-47` and tested by `test_ui_assets_are_served_with_browser_safe_types`. Check with `(Invoke-WebRequest http://127.0.0.1:8790/static/styles.css).Headers['Content-Type']`; expect `text/css; charset=utf-8`. |
| Stuck at "connecting…"; Send does nothing; chips are dead | The script never initialized: a corrupt `kumaru.messages` (line 36 throws), or an id renamed in `index.html` (a `$()` returns null and `init()` throws) | Read the DevTools console. Run `localStorage.removeItem("kumaru.messages")`, or restore the id. |
| Red dot, "server unreachable" | `/v1/info` could not be fetched or parsed (server down or restarting), or something inside `refreshInfo()` threw; one `catch` covers it all | Check the server, then the console |
| Red dot, "load failed", the error shown in the hint | The backend failed to load, and the server does not retry. Chat returns 503 `Backend failed to load: …`. | Fix the cause and restart, or **Load model** a working backend; a successful swap clears the error |
| "loading model…" for minutes | First run downloading weights, or a large load. A chat request waits up to 600 s, then gets 503 `Backend is still loading.` | Wait; watch the server log |
| Think changes nothing | The backend or model has no thinking support: a non-reasoning model, `openai_compat` with `strict_openai`, or in-process `llamacpp` (see the backend table) | Use a reasoning model on `llamacpp_server` (e.g. `-c qwen36`) |
| The model thinks with Think off | In-process `llamacpp` never passes the flag. The model's template default applies, and its inline `<think>` is split out and shown. | Use `llamacpp_server` |
| The budget slider changes nothing | Only `llamacpp_server` (and echo) honour `thinking_budget`. Also, "unlimited" omits the field, so a profile budget still applies. | Pick a stop; check `thinking_mode.thinking_budget` in the profile |
| "No answer: the max-tokens cap ran out while the model was still thinking…" | `finish_reason: "length"` before any answer token: `max_tokens` counts thinking tokens too | Set a thinking budget below max tokens, or untick tuned settings and raise max tokens |
| Max-tokens label reads 3,000 but 4,096 is sent after unticking | The label shows the exact profile value; the knob parks on the nearest stop | Expected; pick a stop |
| Moving a slider changes nothing | **Use the model's tuned settings** is ticked, so slider values are not sent | Untick it |
| **Last response** did not change | Stop or an error: there is no finish frame | Expected |
| tok/s far below the engine's decode speed | A whole-request average including queue wait, prefill and thinking tokens | Read the engine's own log for decode speed |
| The partial answer vanished and "Request failed: …" appeared | An in-band error mid-stream discards what arrived; only Stop keeps a partial reply | Retry. To keep partials, push `finished(…)` before the error row. |
| "Request failed: HTTP 422" | FastAPI validation failed, usually a custom control sending an out-of-range value; the body has `detail`, not `error` | Read the response in the Network tab |
| "Request failed: Invalid or missing API key." | `server.api_key` is set and this file sends no `Authorization` header | See "Send an API key" |
| Every request fails with 503 after **Load model** | The swap closes the old backend first (to free VRAM). A failed load leaves no backend. An empty model box sends `model: ""`, which replaces the configured model; the server drops only `null` fields (`routes_chat.py:253`). | Load a working model, or restart the server |
| The backend list is empty or stale | `<select>` is filled once, on the first poll that returns `available_backends` | Reload the page |
| "copy" does nothing | The Clipboard API needs a secure context: `127.0.0.1` and `localhost` qualify, but a LAN IP over plain HTTP does not, and the handler throws. Or the reply is still streaming: buttons are wired by `renderAll()` only. | Use localhost or HTTPS; wait for the reply |
| History is missing | `localStorage` is per origin (host and port) | Always open the same URL |
| Older replies lost their thoughts after a reload | Over quota: `persist()` saved them without `reasoning` | Expected; shorten the history |
| The next request fails right after an error or a stopped-before-answer reply | Two user turns are now adjacent (empty replies and error rows are not sent). Chat templates that enforce strict user/assistant alternation reject that. | **Clear conversation** |
| A reply lands in a conversation you just cleared | **Clear** does not abort the running stream; the reply is pushed when it finishes | Stop first |
| Enter that picks an IME candidate sends the message | No `isComposing` check (553) | See "Keyboard" above |
| Typing goes nowhere after a reply | `#input` was disabled during the stream and is not refocused | Click it, or apply the focus snippet |
| `<b>` or a table shows literally | Escape-first by design; unsupported syntax | Add a rule (see above) |
| "Thinking… 12s" stops counting | The label is redrawn only when a frame arrives: the engine stalled, or a proxy is buffering | Check the server log and the proxy |
| "60s" or "1m 60s" | `formatSeconds()` rounds after choosing the unit (59.6 s → "60s", 119.6 s → "1m 60s") | Cosmetic |
| The reply appears all at once at the end | A reverse proxy buffers the stream. The server sends `X-Accel-Buffering: no`, which nginx honours; other proxies need buffering turned off. | Configure the proxy |
| Old behaviour after editing this file | Browser cache | Ctrl+F5 |
| Hint says "your data stays on device" with a remote `openai_compat` backend | Hard-coded copy (lines 141, 501; `index.html:108,135`) shown for any backend | Be aware; edit the copy |

## Related files

- `ui\index.html` — the element ids this script binds to
- `ui\styles.css` — the classes it toggles: `.hidden`, `.is-disabled`, `.thinking.live`, `.dot.ok`/`.err`/`.loading`, `.pill[aria-pressed]`
- `src\kumaru\server\app.py` — serves `ui\` at `/` and `/static`, pins MIME types; `ensure_ready()` and `swap_backend()`
- `src\kumaru\server\routes_chat.py` — `/v1/chat/completions` (`_stream_sse`), `/v1/info`, `/v1/backend`
- `src\kumaru\server\schemas.py` — request fields (`thinking`, `thinking_budget`, sampling) and `sse_chunk()`
- `src\kumaru\agent\chat.py` — `resolve_params()`, which `profileDefaults()` mirrors
- `configs\test.yaml` — the echo profile for UI work; `configs\qwen36.yaml` — a profile with a `thinking_mode` preset
- `tests\test_server.py` — pins the contract this parser relies on: frame order, final stats, thinking fields, MIME types
