# `ui\index.html`

> The static 142-line shell of the web UI. It declares every element `ui\app.js` drives, the text shown before the first status poll (or for good, if the script fails), and the load order. No template engine and no build: the server returns this file as-is at `/`.

**Read this when:**
- you add, rename or remove a control, panel, stat or button;
- the page loads unstyled, sits at "connecting…", or says "server unreachable" while the server is fine;
- you change the copy (suggestions, hints, privacy claims), or serve the UI under a path prefix;
- you work on accessibility: labels, names, announcements.

---

## What it does

`ui\index.html` is the whole DOM of the UI. `src\kumaru\server\app.py` mounts the `ui\` folder at `/static` and returns this file for `GET /` (lines 226–231). Both happen only when `server.serve_ui` is true (the default, `src\kumaru\core\config.py:178`) and `ui\` exists. The page:

- loads `/static/styles.css` in `<head>` (line 7) and `/static/app.js` as a classic script at the end of `<body>` (line 140);
- uses an inline SVG data-URI favicon (line 8), so the icon costs no request;
- lays out two columns inside `#app` (line 11):
  - `aside#sidebar` (14–101) holds the brand and version, then the panels **Status**, **Last response**, **Thinking**, **Decoding**, **System prompt** and **Model**, then a footer with **Clear conversation** and **API docs ↗**;
  - `main#main` (104–137) holds `#messages`, which contains a static empty state, then `#composer`: the message box, a bar with the **Think** pill, Send and Stop, and the hint line.

What v0.2 changed (`git --no-pager diff ui\index.html`):

| Change | Lines |
|---|---|
| **Last response** gained a `thinking` row, `#stat-think` | 42 |
| New **Thinking** panel: an explainer and the thinking-budget slider (`#budget-row`, `#val-budget`, `#thinking-budget`) | 48–57 |
| **Decoding** gained **Use the model's tuned settings** (`#auto-sampling`), ticked by default, with an explainer | 61–66 |
| `#max-tokens` became an index slider (`min="0" max="7" value="2"`, label `1,024`); its hint now says "thinking included" | 72–76 |
| The composer gained `.composer-bar`: the **Think** pill (`#think-btn`), a spacer, then Send and Stop | 120–133 |

## Why it exists

The UI must be served by a Python process with no Node toolchain, work offline, and be auditable in one sitting. So every control is declared here once, as static markup; `app.js` fills in everything dynamic; and the only server contract is the public API (`/v1/info`, `/v1/chat/completions`, `/v1/backend`). The decisions that are not obvious from the markup:

| Decision | Where | Why |
|---|---|---|
| All dynamic content is written by `app.js`; the HTML text is the pre-script state | 19, 27, 30–34, 41–44, 106–114, 135 | Before the first `/v1/info` poll (and for good, if the script fails) you see "connecting…" and "—" placeholders. A dead script is then obvious instead of showing plausible stale data. |
| The two token sliders are index sliders | 54, 74 | Their value indexes `BUDGET_STEPS` / `MAX_TOKEN_STEPS` (`app.js:30,33`): hand-picked, roughly logarithmic stops. A linear 256–32,768 range would squeeze the useful low end into a few pixels. |
| The slider defaults equal `configs\default.yaml`'s `generation` block: temperature 0.7, max tokens 1,024, top-p 0.95 | 69, 74, 79 | Before the first poll, the labels already show what the default profile uses. After it, `app.js` parks the sliders on the live profile's values. |
| The budget slider defaults to its last stop: `value="8"` is unlimited | 53–54 | `BUDGET_STEPS[8]` is `null`, so by default the request carries no `thinking_budget`. |
| **Think** is a pill in the composer, not a sidebar setting | 121–125 | Thinking trades speed for quality, so it is chosen per message (the comment at `styles.css:224` says the same). It also stays reachable below 860 px, where the sidebar is hidden. |
| The pill is a `<button type="button">` with `aria-pressed` | 121 | A native toggle: Space and Enter work, and screen readers announce pressed or not pressed. `styles.css:229` styles the pressed state from the same attribute, so state and look cannot drift apart. |
| `#swap-backend` ships with no `<option>`s | 92 | It is filled from the server's registry (`available_backends`) on the first poll, so the list cannot go stale against the server. |
| `#stop-btn` ships with `class="hidden"` | 130 | The idle state is correct before `app.js` runs. `setStreaming()` swaps Stop and Send from then on. |
| Inline SVG icons and favicon | 8, 123, 128, 131 | No icon font and no extra requests: nothing to fetch offline. |
| Absolute URLs (`/static/…`, `/docs`, and `/v1/…` in `app.js`) | 7, 99, 140 | The simplest correct choice when the UI lives at the origin root. It is also why a path-prefixed deployment breaks (see Failure modes). |
| A classic script at the end of `<body>`, started on `DOMContentLoaded` (`app.js:604`) | 140 | The tag works there, in `<head>`, or with `defer`. Never add `async`: an async script can run after `DOMContentLoaded` has fired, and then `init()` never runs. |

## Mental model (DOM contract)

Three files share names, and nothing checks them: no build, no types, no browser test. The suite only checks that `/`, `/static/app.js` and `/static/styles.css` are served with the right content types.

1. **Ids are `app.js`'s contract.** The file declares 38 ids, and `app.js` looks up 33 of them by name with `$()`. The other five exist for layout only: `#app`, `#sidebar`, `#main` and `#composer` are styled by id, and `#model-facts` is used by neither file (`.kv` styles it).
2. **Four classes are part of `app.js`'s contract too, not just styling.**
   - `.slider`: `syncControls()` calls `$(id).closest(".slider")` for the three sampling sliders (`app.js:458`). Remove the wrapper and it throws.
   - `.chip`: `wireChips()` attaches its click handler to every `.chip` in the document (`app.js:188-195`). Any element with that class sends its own text as a message.
   - `.hidden`: `setStreaming()` toggles it on Send and Stop.
   - `dot`: `refreshInfo()` replaces the whole `className` of `#status-dot` with `dot ok`, `dot loading` or `dot err`, so any other class put there is wiped on the first poll.
3. **The static text is a placeholder, not the source of truth.** `app.js` rewrites `#version`, the status, the facts, the stats, every `val-*` label and `#hint`. It also rewrites all of `#messages`: `renderAll()` replaces the empty state at init (`app.js:598`) with its own copy (`app.js:139-146`).
4. **Control values here are only starting points.** From init on, `syncControls()` (`app.js:454-482`) owns the disabled states, `.is-disabled`, the labels and `aria-pressed`. `refreshInfo()` sets the first Think state from `generation.thinking === true`. Nothing a user sets here survives a reload; only the conversation is persisted.

### Which function reads which id

A renamed or deleted id fails where it is first read, and that decides the symptom:

| Ids | Read by | If one is missing |
|---|---|---|
| `input`, `send-btn`, `stop-btn`, `clear-btn`, `swap-btn`, `think-btn`, `temperature`, `max-tokens`, `top-p`, `thinking-budget`, `auto-sampling`, `val-temp`, `val-max`, `val-topp`, `val-budget`, `budget-row`, `messages` | `init()` (`app.js:545-598`), directly or through its first `syncControls()` (573) and `renderAll()` (598) | `init()` throws before `refreshInfo()` (599). The page stays at "connecting…", nothing after the failing line is wired, and the static chips are dead. |
| `version`, `hint`, `fact-model`, `fact-backend`, `fact-device`, `fact-dtype`, `fact-context`, `swap-backend` | `refreshInfo()` (`app.js:486-540`) | The write throws inside its `try`, and the `catch` (536–538) shows a red dot and "server unreachable" while the server is fine. |
| `status-dot`, `status-text` | `refreshInfo()` and its `catch` | Both paths throw; the error shows only in the console. |
| `swap-model` | `refreshInfo()` on the first poll only (526), and the swap handler (583) | "server unreachable" for one poll, then normal. **Load model** alerts a `TypeError`. |
| `system-prompt` | `send()` (223) | The message is shown, saved and cleared from the box, then nothing happens: no reply, no error row. The rejection appears only in the console. |
| `stat-ttft`, `stat-think`, `stat-tps`, `stat-tokens` | `updateStats()` (`app.js:422-427`), on the final frame | The reply streams, then is replaced by an error row "Request failed: Cannot set properties of null (setting 'textContent')" (Chromium's wording). |

### Slider reference

| Slider | Line | Range in HTML | What the value means | Request field |
|---|---|---|---|---|
| `#thinking-budget` | 54 | 0–8, step 1, value 8 | index into `BUDGET_STEPS`: 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, 24,576, unlimited | `thinking_budget`, only with Think on and not at unlimited |
| `#temperature` | 69 | 0–1.5, step 0.05, value 0.7 | the value itself | `temperature`, only with tuned settings unticked |
| `#max-tokens` | 74 | 0–7, step 1, value 2 | index into `MAX_TOKEN_STEPS`: 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, 32,768 | `max_tokens`, only with tuned settings unticked |
| `#top-p` | 79 | 0.1–1, step 0.01, value 0.95 | the value itself | `top_p`, only with tuned settings unticked |

Three rules follow:

- **An index slider's `max` must be its array's length − 1.**
  - One stop too many on `#max-tokens` makes `MAX_TOKEN_STEPS[8]` undefined. With tuned settings unticked, `syncControls()` then throws at `maxTok.toLocaleString()` (`app.js:474`) on every input and every poll: the label freezes, the status flips to "server unreachable", and the request silently omits `max_tokens`.
  - One stop too many on `#thinking-budget` fails silently: `undefined == null`, so it shows and behaves as "unlimited".
- **`value` is an index too.** To change a default, change `value` to the index of the stop you want; never type a token count.
- **The plain ranges are narrower than the API.** The server accepts temperature up to 2.0 and top-p down to 0.0 (`src\kumaru\server\schemas.py:60-61`). A profile value outside the slider's range still applies while tuned settings are ticked (nothing is sent, and the label shows the profile's value). But the knob is clamped, so unticking sends the clamped value.

### Copy that makes claims

| Text | Here | Also in `app.js` |
|---|---|---|
| "Running entirely on your machine. Nothing leaves this computer." | 108 | 141, the copy actually shown |
| "Local model · your data stays on device" | 135 | 501, rewritten on every poll that reports ready |
| "First run downloads weights — this can take several minutes." | — | 505, while the model loads |
| "Swaps in place. Frees VRAM before loading." | 94 | — |

- The first two lines are hard-coded for every backend. With `openai_compat` pointing at a remote API, messages do leave the machine.
- "Frees VRAM before loading" is literal: the server closes the old backend before loading the new one, so a failed load leaves no backend at all.

## Structure walkthrough

| Line | Section / id | Purpose | JavaScript dependency |
|---|---|---|---|
| 1–2 | `<!DOCTYPE html>`, `<html lang="en">` | Standards mode, which is what makes browsers refuse a mistyped stylesheet. `lang` sets the screen-reader language. | — |
| 4–5 | `charset`, `viewport` (`width=device-width, initial-scale=1`) | Without the viewport tag, phones lay the page out about 980 px wide and the ≤ 860 px rule in `styles.css` never fires. | — |
| 6 | `<title>Kumaru</title>` | Tab title | never changed |
| 7 | `<link rel="stylesheet" href="/static/styles.css">` | Render-blocking in `<head>`, so there is no flash of unstyled content | — |
| 8 | Data-URI SVG favicon `◈` | No request | — |
| 11 | `#app` | The two-column grid | — |
| 14 | `aside#sidebar` | Status and controls; hidden at ≤ 860 px | — |
| 15–21 | `header.brand`: `.mark`, `h1`, `#version` ("local agent") | Brand and version | `refreshInfo()` writes `v` + `info.version` (516). Today that is `v0.2.0` (`__version__`, `src\kumaru\__init__.py:16`). |
| 23–28 | **Status**: `#status-dot` (`dot loading`), `#status-text` ("connecting…") | Server and model state | `refreshInfo()` sets `dot ok` "ready", `dot loading` "loading model…" or `dot err` "load failed" (494–506), and `dot err` "server unreachable" when the poll fails (536–538). |
| 29–35 | `dl.kv#model-facts`: `#fact-model`, `#fact-backend`, `#fact-device`, `#fact-dtype`, `#fact-context` | What is actually loaded | `refreshInfo()` (510–515) prefers the live `info.agent.backend`, then `info.config.backend`, else "—". Context comes only from the live backend. `#fact-model` also gets a `title`, because `.kv dd` truncates with an ellipsis. |
| 38–46 | **Last response**: `#stat-ttft`, `#stat-think`, `#stat-tps`, `#stat-tokens` | Stats of the latest finished reply | `updateStats()` (422–427), from the final frame's `kumaru` object: `N ms`; `formatSeconds()` or `off`; `N tok/s`; `prompt → completion` |
| 48–51 | **Thinking** explainer | Says where Think is and what it costs | — |
| 52–56 | `label.slider#budget-row`: `#val-budget`, `#thinking-budget`, hint | Cap on thinking tokens | `syncControls()` writes the label (476–477), and disables the slider and adds `.is-disabled` to the row while Think is off (478–479). `requestSettings()` sends `thinking_budget` (403–405). |
| 61–66 | `label.check`: `#auto-sampling` (checked), plus a muted explainer | Lets the profile decide sampling | While ticked, `syncControls()` disables the three sliders, adds `.is-disabled` and parks them on the profile's values (455–468), and `requestSettings()` omits `temperature`, `top_p` and `max_tokens` (407–411). |
| 67–71 | `#temperature`, `#val-temp` | Sampling temperature | sent as `temperature` while unticked |
| 72–76 | `#max-tokens`, `#val-max` | Reply cap, thinking included | `MAX_TOKEN_STEPS[value]`, sent as `max_tokens` while unticked |
| 77–81 | `#top-p`, `#val-topp` | Nucleus sampling | sent as `top_p` while unticked |
| 84–87 | `#system-prompt` | Extra instructions for this tab | When non-empty, `send()` puts it first as a `system` message (223–224). The server appends it to its own prompt (`ensure_system_message`, `src\kumaru\agent\prompt.py:64`), so it cannot remove server-side rules. Not persisted. |
| 89–95 | **Model**: `#swap-model`, `#swap-backend`, `#swap-btn`, note | Hot-swap model or backend | On the first poll, `refreshInfo()` fills the select from `available_backends`, preselects the current backend and prefills the model box with `config.backend.model` (518–527). The button posts `{name, model}` to `/v1/backend` and alerts on failure (575–596). |
| 97–100 | `footer.side-footer`: `#clear-btn`, `/docs` link | Reset; API reference | `#clear-btn` empties `state.messages`, saves and re-renders (559–563). It does not abort a running stream. The link opens FastAPI's Swagger UI in a new tab. |
| 104 | `main#main` | The conversation column | — |
| 105 | `#messages` | The scroll container for the conversation | `renderAll()` rebuilds it; `send()` appends the live reply; `isNearBottom()` and `scrollToBottom()` measure and scroll it. |
| 106–114 | Static `.empty-state` and three `.chip`s | Fallback text only | Replaced by `renderAll()` at init. The static chips never get a handler. |
| 117–118 | `#composer`, `.composer-inner` | Composer frame, highlighted on `:focus-within` | — |
| 119 | `textarea#input` | The message box | Grows up to 200 px (547–550). Enter sends and Shift+Enter adds a newline (552–554). Disabled while streaming (419); focused at init (601). |
| 120 | `.composer-bar` | Flex row under the box | — |
| 121–125 | `#think-btn.pill` (`aria-pressed="false"`) | The Think toggle | A click flips `state.think` and marks it user-chosen (568–572). `syncControls()` writes `aria-pressed` (481). The first poll with a `config` sets the starting state from `generation.thinking === true`, unless the user clicked first (531–534). |
| 126 | `span.spacer` | Pushes Send and Stop to the right (`flex: 1`) | — |
| 127–129 | `#send-btn` (`title="Send"`) | Send | calls `send()`; hidden while streaming |
| 130–132 | `#stop-btn.hidden` (`title="Stop"`) | Stop | aborts `state.controller` (557); shown only while streaming |
| 135 | `p.hint#hint` | One line of status under the composer | `refreshInfo()` rewrites it on every poll: the ready text, the loading text, or `info.error` (497–505) |
| 140 | `<script src="/static/app.js">` | Client behaviour | `init()` runs on `DOMContentLoaded` (`app.js:604`) |

## How to customize it

### Change the suggestions

The chips users see come from `renderAll()` (`app.js:143-145`), which replaces the static empty state at init and after **Clear conversation**. Edit them there. Mirror the change at lines 110–112 only to keep the no-script fallback consistent. A chip sends its visible text verbatim (`wireChips()` copies `textContent` into `#input`), so the label is the prompt.

### Add a control that changes requests

There are four touch points, plus the server:

1. **Markup here**, in the right panel. Wrap a slider in `<label class="slider">` with a `<span>` label, a `<b id="val-…">` value and a `<small>` hint; `.is-disabled` and the label styles depend on that shape.
2. **`requestSettings()`**: what to send, and in which state.
3. **`syncControls()`**: the label, the disabled state, and parking on the profile value.
4. **`init()`**: add the id to the `input` listener list (`app.js:565`).
5. **The server schema** (`src\kumaru\server\schemas.py`). It uses `extra="ignore"`, so an unknown field returns 200 and changes nothing.

`docs\howto\ui\app.js.md` ("Add a request parameter (worked example: `top_k`)") walks through the full example, including a test that proves the value reaches the backend.

### Add or remove a slider stop

Edit the array in `app.js` (`MAX_TOKEN_STEPS` at line 30, `BUDGET_STEPS` at 33) and this file's `max` in the same change. Then re-check `value`, which is an index. For example, appending 65,536 to `MAX_TOKEN_STEPS` needs `max="8"` on line 74.

### Add a backend to the Model list

Do not add `<option>`s. `refreshInfo()` fills `#swap-backend` from `available_backends`, which is `BACKENDS.names()` (`src\kumaru\server\routes_chat.py:233`): today `echo`, `llamacpp`, `llamacpp_server`, `openai_compat` and `transformers`. Registry aliases are not listed. The list is filled only while the select is empty, so register the backend server-side, restart the server, and reload the page.

### Fix the accessible names

Today, `#swap-backend` has no name at all. `#system-prompt`, `#swap-model` and `#input` are named only by their placeholders, which browsers use as a last resort and which disappear visually once the user types. Send and Stop are icon-only and named only by `title`. Give each one an explicit name:

```html
<textarea id="system-prompt" rows="4" aria-label="System prompt" placeholder="Optional. Added on top of the server's prompt."></textarea>
<input type="text" id="swap-model" aria-label="Model" placeholder="Qwen/Qwen2.5-7B-Instruct" />
<select id="swap-backend" aria-label="Backend"></select>
<textarea id="input" rows="1" aria-label="Message" placeholder="Message Kumaru…  (Enter to send, Shift+Enter for newline)"></textarea>
<button id="send-btn" title="Send" aria-label="Send">…</button>
<button id="stop-btn" class="hidden" title="Stop" aria-label="Stop">…</button>
```

The two index sliders report their index (for example "8"), not "unlimited". Give them a spoken value in `syncControls()`:

```js
$("max-tokens").setAttribute("aria-valuetext", `${maxTok.toLocaleString()} tokens`);
$("thinking-budget").setAttribute("aria-valuetext", budget == null ? "unlimited" : `${budget.toLocaleString()} tokens`);
```

Nothing is announced: there is no live region. If you add one, write to it once per event (reply finished, model loaded or failed). Do not make `#messages` live: it is rewritten on every animation frame while streaming. Do not make `#status-text` or `#hint` live either until `refreshInfo()` skips identical writes, because it rewrites them every 4 s.

### Serve the UI under a path prefix

Six URLs are absolute:
- in this file: `/static/styles.css` (7), `/docs` (99), `/static/app.js` (140);
- in `app.js`: `/v1/chat/completions` (292), `/v1/info` (488), `/v1/backend` (580).

Behind a reverse proxy at a sub-path, they escape the prefix. Make all six relative (`static/app.js`, `v1/info`, …) and open the UI with a trailing slash (`/kumaru/`, not `/kumaru`), so they resolve under the prefix. The Swagger page also loads `/openapi.json` by absolute path. FastAPI's `root_path` handles that, and `create_app()` does not set it (`app.py:191-196`).

### Change or remove the API docs link

`/docs` is FastAPI's default Swagger UI, and that page loads its JavaScript and CSS from `cdn.jsdelivr.net`. On an offline machine it is blank. `/openapi.json` is generated locally and always works, so point the link there on offline installs, or remove it.

### Change the privacy copy

Edit both copies (lines 108 and 135 here, `app.js:141` and `501`). To make the claim honest automatically, derive it in `refreshInfo()` from `info.config.backend`. `name` alone is not enough, because `openai_compat` can point at a local server too; check whether `base_url` is a loopback address.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Unstyled page. Chromium's console says "Refused to apply style from '…/static/styles.css' because its MIME type ('text/plain') is not a supported stylesheet MIME type, and strict MIME checking is enabled." | Python's `mimetypes` reads the Windows registry, where some installers map `.css` and `.js` to `text/plain`. Line 1 puts the page in standards mode, where browsers refuse such a stylesheet. | Fixed in v0.2: `app.py:46-47` pins `text/css` and `text/javascript`, and `test_ui_assets_are_served_with_browser_safe_types` guards it. With `kumaru serve -c test --port 8790` running, `(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8790/static/styles.css).Headers['Content-Type']` must print `text/css; charset=utf-8`. |
| Stuck at "connecting…" with a pulsing amber dot; chips are dead; Send does nothing | `app.js` never ran, or `init()` threw before `refreshInfo()`. Causes: a 404 on `/static/app.js`, a syntax error, corrupt `localStorage["kumaru.messages"]` (`app.js:36` throws), `async` added to line 140, or a renamed id (see "Which function reads which id") | Read the DevTools console. Restore the id or the tag. For corrupt storage, run `localStorage.removeItem("kumaru.messages")`. |
| Red dot and "server unreachable" while the API answers | An id read only by `refreshInfo()` is gone, or an index slider's `max` runs past its array while tuned settings are unticked | The console shows the `TypeError`. Restore the id, or fix `max`. |
| You edited the chips here and nothing changed | `renderAll()` replaces the static empty state at init | Edit `app.js:143-145`. |
| Clicking some other button sends its label as a message | It has class `chip`, and `wireChips()` wires every `.chip` in the document | Use another class name. |
| A class added to `#status-dot` disappears | `refreshInfo()` overwrites its `className` on every poll | Style it from the parent, or use a data attribute. |
| The max-tokens label freezes after you widen the slider | `max` above 7 makes `MAX_TOKEN_STEPS[i]` undefined, and `syncControls()` throws (`app.js:474`) | Keep `max` at the array length − 1. |
| The budget slider shows "unlimited" at a stop you just added | `max` runs past `BUDGET_STEPS` (an undefined entry reads as unlimited), or the stop was added here but not in `app.js:33` | Change both together. |
| The sampling sliders are greyed out and do not move | **Use the model's tuned settings** is ticked (the default): the profile decides, and the sliders are inert | Untick it. |
| The budget slider is greyed out | Think is off; the budget only applies to thinking replies | Turn Think on. |
| Every setting resets on reload | Only the conversation is persisted. Controls return to their HTML defaults, and Think to the profile default. | Expected. See `docs\howto\ui\app.js.md`, "Remember the Think choice across reloads". |
| `GET /` returns 404 `{"detail":"Not Found"}` | `server.serve_ui` is false, or there is no `ui\` at `REPO_ROOT\ui` (`config.py:34`, `app.py:41`). Only a source checkout has one: `pyproject.toml` ships only `py.typed` as package data, so a non-editable install has no UI. | Run from the repo with `pip install -e .`, and keep `serve_ui: true`. |
| Blank page and 404s for `/static/…` or `/v1/…` behind a proxy sub-path | The absolute URLs bypass the prefix | See "Serve the UI under a path prefix". |
| **API docs ↗** opens a blank page | Offline: FastAPI's Swagger UI loads its assets from `cdn.jsdelivr.net` | Use `/openapi.json`, which is generated locally. |
| Edits to this file, the CSS or the JS do not show | The browser reused a cached copy. Static files carry `ETag` and `Last-Modified` but no `Cache-Control`, so freshness is heuristic. | Ctrl+F5, or DevTools → Network → Disable cache. |
| The sidebar is gone | The window is ≤ 860 px wide, and `styles.css:240-243` hides the sidebar. Think stays usable because it is in the composer; the budget, sampling, system prompt and model swap are unreachable. | Widen the window, or see `docs\howto\ui\styles.css.md`, "Make narrow screens usable". |
| Every chat returns 503 after a failed **Load model** | The swap frees VRAM first: it closes the old backend, then loads, so a failed load leaves no backend. An empty model box sends `model: ""`, which replaces the configured model. | Load a working model, or restart the server. |
| A screen reader says "8" for the budget or "2" for max tokens; the backend select has no name; nothing is announced when a reply finishes | Index sliders without `aria-valuetext`, an unlabelled `#swap-backend`, no live region | See "Fix the accessible names". |
| The page claims "your data stays on device" with a remote backend | Hard-coded copy | See "Change the privacy copy". |

## Related files

- `ui\app.js` — reads 33 of this file's 38 ids and owns all dynamic content; see `docs\howto\ui\app.js.md`
- `ui\styles.css` — styles `#app`, `#sidebar`, `#main`, `#messages`, `#composer`, `#input`, `#send-btn`, `#stop-btn` and the classes used here; see `docs\howto\ui\styles.css.md`
- `src\kumaru\server\app.py` — serves this file at `/` and `ui\` at `/static` (226–231); pins the MIME types (43–47)
- `src\kumaru\server\routes_chat.py` — `/v1/info` (224–245), `/v1/chat/completions`, `/v1/backend` (248–259)
- `src\kumaru\core\config.py` — `REPO_ROOT` (34) and `server.serve_ui` (178)
- `configs\default.yaml` — the `generation` values that the slider defaults mirror
- `tests\test_server.py` — `test_ui_assets_are_served_with_browser_safe_types` (183), the only automated check on the UI files
