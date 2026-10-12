# `ui\styles.css`

> The UI's only stylesheet: 243 hand-written lines with no framework, preprocessor or build step. It defines one dark theme from 16 custom properties and a two-column grid that never lets the conversation push the composer off-screen. It is also the visual half of every state that `ui\app.js` toggles.

**Read this when:**
- you change the theme, the layout or the breakpoint;
- you add a UI state, a message role or a markdown element and need the class contract with `app.js`;
- the page renders unstyled, the composer slides below the window, Send and Stop both show, or keyboard focus is invisible.

---

## What it does

`ui\styles.css` styles two kinds of markup:

- the static shell in `ui\index.html`: sidebar panels, controls, the composer, the Think pill;
- the markup `ui\app.js` builds at runtime: message rows (`messageEl()`), markdown (`renderMarkdown()`), thinking panels (`thinkingHtml()` and the live panel in `send()`), the streaming cursor, notes and the stats line.

`index.html` loads it render-blocking from `<head>` (line 7). The server serves it from the repo's `ui\` folder at `/static/styles.css`, with the type pinned to `text/css` (`src\kumaru\server\app.py:43-47`), because browsers ignore a stylesheet served as `text/plain`.

What v0.2 changed (`git --no-pager diff ui\styles.css`):

| Change | Lines |
|---|---|
| `#app` gained `grid-template-rows: minmax(0, 1fr)`: a long conversation no longer pushes the composer below the window | 38–40 |
| An inert style for disabled sliders (`.slider.is-disabled`, `input[type="range"]:disabled`), used by **Use the model's tuned settings** and the budget slider | 87–90 |
| `.check` (the tuned-settings checkbox) and `.muted b` | 92–94 |
| `.bubble .note`, the italic note shown when an answer is empty | 187 |
| The thinking panel: `.thinking`, its `summary` and chevron, `.live`, `.thinking-body` | 189–208 |
| `.composer-inner` became a column: the message box on top, `.composer-bar` (Think, spacer, Send and Stop) below | 213–217 |
| `#input` padding adjusted to the new layout | 219 |
| `.pill`, the Think toggle, styled from `aria-pressed` | 224–231 |

## Why it exists

The whole UI is three static files that the server hands out as they are (the header comment, lines 1–4). There is no npm, no bundler and no version drift. Every rule that affects the screen is readable in one file, and a change needs only a refresh. The decisions that are not obvious from the rules:

| Decision | Where | Why |
|---|---|---|
| `grid-template-rows: minmax(0, 1fr)` on `#app` | 38–40 | It makes the row exactly the window's height. An `auto` row (and a bare `1fr`, which means `minmax(auto, 1fr)`) grows with the conversation and pushes the composer off-screen; this was the v0.1 behaviour. |
| `min-width: 0` on `#main` and `.bubble` | 137, 156 | Grid and flex items default to `min-width: auto`, which resolves to their min-content width. Without it, one long code line widens the whole column instead of scrolling inside its `pre` (`overflow-x: auto`, 168–169). |
| `.hidden { display: none !important; }` | 233 | It must beat `#send-btn, #stop-btn { display: grid; }` (221). An id selector outranks a class, so without `!important` Send and Stop would both show all the time. |
| Disabled sliders stay visible at half opacity | 87–90 | While the profile controls sampling, the sliders show the profile's values ("visible, but inert", line 87). Hiding them would hide what the model actually uses. |
| The thinking body is plain text in a fixed-height scroller | 189–191, 206–208 | A 20,000-token thought never shoves the answer off-screen. `white-space: pre-wrap` keeps the model's line breaks without parsing anything; thoughts are a text node, never markdown. `overflow-wrap: anywhere` stops a long token widening the bubble. |
| Collapsed thinking shrinks to `fit-content`; the chevron is drawn with borders | 197–204 | Closed, the panel is just a label. The drawn chevron is "crisp at any zoom, unlike a tiny ▸ glyph" (line 197). The native markers are suppressed by `list-style: none` and `::-webkit-details-marker`. |
| The pressed Think pill is styled from `[aria-pressed="true"]` | 229–230 | The ARIA state that screen readers announce is the same state the eye sees, so the two cannot drift apart. |
| Data in `--mono`, prose in `--sans` | 21–22, 218 | The version line, model facts, slider values, form fields, code and the stats line use the monospace stack, so ids and numbers line up and read as data. The message box switches to `--sans` at 14 px because it holds prose. |
| The copy button is hidden until the code block is hovered | 177–180 | Code blocks stay clean while reading. The cost: the button is invisible to keyboard and touch users. |
| The sidebar is hidden at ≤ 860 px | 240–243 | The conversation keeps the full width on phones. The Think pill lives in the composer, not the sidebar, so thinking stays reachable. |

## Mental model

There are four layers:

1. **Tokens** (`:root`, 6–23): 16 custom properties, made of 13 colours, `--radius` and two font stacks. Most visual changes start here. Six colours are hard-coded and ignore the tokens; they are listed under "Change the theme colours".
2. **Layout** (25–40, 44–52, 137–139, 212–222, 240–243). `#app` is a two-column grid exactly one window tall, and `#sidebar` scrolls on its own. `#main` is a flex column: `#messages` takes the leftover height and scrolls, and `#composer` stays at the bottom. At ≤ 860 px the grid collapses to one column and the sidebar is hidden.
3. **Components**: sidebar panels and controls, buttons, message rows, markdown, the thinking panel, the composer and the pill.
4. **State hooks**: the classes and attributes that `app.js` sets at runtime. They are the contract between the two files, and nothing checks them. A rename on one side fails silently: the element loses its style, or never hides.

| Hook | Set by (`ui\app.js`) | Effect here |
|---|---|---|
| `.hidden` | `setStreaming()` swaps it between Send and Stop (417–418). The live thinking panel is created with it (238–242) and loses it at the first thought (276). | `display: none !important` (233) |
| `.slider.is-disabled`, `input[type="range"]:disabled` | `syncControls()`: on the three Decoding sliders (temperature, top-p, max tokens) while **Use the model's tuned settings** is ticked (456–458), and on the budget row while Think is off (478–479) | half opacity, a dimmed value, a not-allowed cursor (88–90) |
| `.dot.ok`, `.dot.loading`, `.dot.err` | `refreshInfo()` overwrites `#status-dot`'s whole `className` (495, 499, 503, 537) | green glow, pulsing amber, red glow (72–74) |
| `.thinking[open]` | The user's clicks. `send()` creates the live panel open and closes it when the answer starts (340). Finished messages render closed. | chevron rotation (201); collapsed width `fit-content` (204) |
| `.thinking.live` | Added by `send()` (238–242); removed when the answer starts (341) | the label pulses (205) |
| `.pill[aria-pressed="true"]` | `syncControls()` (481) | `--accent` text, an `--accent-dim` border and a 9% accent tint (229–230) |
| `.msg.user`, `.msg.assistant`, `.msg.error` | `messageEl()` sets `class="msg <role>"` (127) | avatar colours (152–154); red error text (157) |
| `button:disabled` | the **Load model** button while a swap runs (577) | faded, not-allowed cursor (123) |

The colour pairs that matter, as WCAG 2 contrast ratios (computed from the token values):

| Pair | Where it is seen | Ratio | Verdict |
|---|---|---|---|
| `--text` on `--bg-panel` | sidebar text | 15.16 | passes AA text (4.5:1) |
| `--text-dim` on `--bg-panel` | fact values, slider labels | 6.10 | passes |
| `--accent` on `--bg-panel` | slider values; the pressed Think pill (about 6.7 on its tint) | 7.64 | passes |
| `#06101f` on `--accent` | primary button labels | 7.89 | passes |
| `--err` on `--bg` | error messages | 5.77 | passes |
| `#fff` on `--err` | the Stop icon | 3.37 | passes the 3:1 bar for non-text UI |
| `--text-faint` on `--bg` / `--bg-panel` / `--bg-elevated` | the small print: panel titles, fact keys, slider notes, hints, the stats line, the copy button, the unpressed Think pill (mostly 10–12 px) | 3.34 / 3.17 / 2.97 | **fails** AA text (4.5:1) |
| `--accent-dim` against `--border` / `--bg-input` / `--bg-panel` / `--bg` | focus borders (114, 215) and the Think focus ring (231) | 1.59 / 1.84 / 2.09 / 2.21 | **fails** the 3:1 bar for focus indicators |

## Structure walkthrough

| Lines | Block | Purpose | Notes |
|---|---|---|---|
| 1–4 | Header comment | Why there is no framework or build | Keep it true if you add tooling. |
| 6–23 | `:root` | The 16 tokens | See "Change the theme colours". |
| 25 | `*` | `box-sizing: border-box`; zero margins and padding | No reset library |
| 27–36 | `html, body`, `body` | Full-height canvas; 14 px / 1.6 `--sans` on `--bg` | |
| 38–40 | `#app` | Two-column grid: a 300 px sidebar, and one row exactly `100vh` tall | `minmax(0, 1fr)` is load-bearing (comment 38–39). |
| 44–52 | `#sidebar` | Panel column that scrolls on its own (`overflow-y: auto`) | `gap: 18px` spaces the panels. |
| 54–57 | `.brand` | Mark, title and version line | `#version` text comes from `app.js`. |
| 59–67 | `.panel`, `.panel-title` | Panel stack, small uppercase titles | |
| 69–76 | `.status-row`, `.dot.*`, `@keyframes pulse` | The status light | `pulse` is reused by the live thinking label (205). |
| 78–81 | `.kv` | Model facts as a two-column grid | Values ellipsize; `#fact-model` also gets the full id as a tooltip (`app.js:511`). |
| 83–90 | `.slider`, `.is-disabled`, `:disabled` | Label row with a `<b>` value and a `<small>` hint; the inert state | `index.html` must keep this shape. |
| 92–94 | `.check`, `.muted b` | The tuned-settings checkbox (native, tinted with `accent-color`); the bold **Think** in the Thinking panel's note (`index.html:50`) | |
| 96–101 | `input[type="range"]` and its thumbs | 3 px track, 13 px accent thumb | WebKit and Firefox thumbs are separate rules. `outline: none` is at 97. |
| 103–114 | `textarea`, `input[type="text"]`, `select`, and their `:focus` | Shared field look (`--mono`, `--bg-input`); focus is shown by an `--accent-dim` border | `resize: vertical` (112) also reaches `#input`. Line 114 strips the outline from every `input`, including ranges and the checkbox. |
| 116–126 | `button`, `.secondary`, `.ghost` | Accent buttons with dark text; hover brightens; disabled fades | No focus rule, so buttons keep the browser's focus ring. |
| 128 | `.muted` | 10 px explanatory copy | |
| 130–133 | `.side-footer` | **Clear conversation** and the **API docs** link, pushed down by `margin-top: auto` | |
| 137 | `#main` | Conversation column (flex) | `min-width: 0` |
| 139 | `#messages` | The scroller (`flex: 1; overflow-y: auto`) | `scroll-behavior: smooth` does not affect `scrollToBottom()`, the app's only programmatic scroll here: it passes `behavior: "instant"` (`app.js:164`), because "a smooth scroll restarted 60 times a second never catches up" (`app.js:162`). |
| 141–147 | `.empty-state`, `.suggestions`, `.chip` | The empty conversation and its suggestion buttons | Static in `index.html`, then re-rendered by `renderAll()` (`app.js:139-146`) |
| 149–157 | `.msg`, `.avatar`, role modifiers, `.bubble` | A 780 px message column; one avatar colour per role; red error text | |
| 159–175 | Markdown output | `p`, lists, `h1`–`h3`, inline and block code, quotes, links, `strong`, tables | The table rules (174–175) never apply: the renderer produces no tables. |
| 177–180 | `.copy-btn` | Copy button inside code blocks, shown on `pre:hover` | No focus or touch state |
| 182–184 | `.cursor`, `@keyframes blink` | The streaming cursor | Only in the live reply |
| 186–187 | `.msg-meta`, `.bubble .note` | The stats line under a reply; the italic note for an empty answer | |
| 189–208 | `.thinking`, `.thinking-body` | Collapsible thoughts above the answer | A native `<details>`: Enter and Space toggle it, and screen readers announce expanded or collapsed. |
| 212 | `#composer` | The bottom strip | |
| 213–217 | `.composer-inner`, `.composer-bar`, `.spacer` | The box on top, the button bar below; a focus-within border | |
| 218–220 | `#input` | Borderless, `--sans` 14 px, at most 200 px tall | The JS height cap is also 200 (`app.js:549`). |
| 221–222 | `#send-btn`, `#stop-btn` | 33 px icon buttons; Stop is red | `display: grid` is why `.hidden` needs `!important`. |
| 224–231 | `.pill` | The Think toggle; pressed state from `aria-pressed`; its own focus ring | The pressed tint is `--accent`, hard-coded. |
| 232 | `.hint` | The status line under the composer | `refreshInfo()` writes the load error, the first-run download notice or the privacy line into it (`app.js:497-505`). |
| 233 | `.hidden` | Hide utility | `!important` is load-bearing. |
| 235–238 | `::-webkit-scrollbar*` | Thin dark scrollbars | Chromium and Safari only |
| 240–243 | `@media (max-width: 860px)` | One column; sidebar hidden | |

## How to customize it

### Change the theme colours

Change the tokens, not individual rules:

```css
:root {
  --bg: #101014;
  --bg-panel: #181820;
  --accent: #b48cff;
  --accent-dim: #4b376d;
}
```

Then fix what the tokens do not reach. Six colours are hard-coded:

| Line | Rule | Colour | Role |
|---|---|---|---|
| 117 | `button` | `#06101f` | dark label on accent buttons |
| 154 | `.msg.error .avatar` | `#3a1416` | error avatar background |
| 167 | `.bubble code` | `#e3c2a0` | inline code text |
| 173, 222 | `.bubble strong`, `#stop-btn` | `#fff` | bold text; the Stop icon |
| 230 | `.pill[aria-pressed="true"]` | `rgba(110, 168, 254, .09)` | the old `--accent` (`#6ea8fe`) at 9%. Replace it with `color-mix(in srgb, var(--accent) 9%, transparent)` so it follows the token. |
| 238 | `::-webkit-scrollbar-thumb:hover` | `#364150` | scrollbar thumb on hover |

Re-check the contrast table in "Mental model" after any change. The theme is dark-only: there is no `prefers-color-scheme` query and no `color-scheme` declaration. Native parts the stylesheet does not restyle, such as the unchecked checkbox, therefore use the browser's light defaults. Add `color-scheme: dark;` to `:root` to make them match.

### Make the sidebar wider

Change the first column on line 40 and keep the rest of the declaration:

```css
#app { display: grid; grid-template-columns: 360px 1fr; grid-template-rows: minmax(0, 1fr); height: 100vh; }
```

Never drop `grid-template-rows: minmax(0, 1fr)`. Without it, a long conversation pushes the composer below the window. The media query (241) still collapses to one column at ≤ 860 px. Long model ids keep ellipsizing in `.kv dd`.

### Make narrow screens usable

At ≤ 860 px the sidebar is `display: none` (240–243). That hides the status light, the model facts, the last-response stats, the thinking budget, the Decoding controls, the system prompt, the model swap and **Clear conversation**. Only Think, in the composer, remains. To stack the sidebar above the conversation instead, replace the media query:

```css
@media (max-width: 860px) {
  #app { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
  #sidebar { display: flex; max-height: 45vh; border-right: none; border-bottom: 1px solid var(--border); }
}
```

- The second row must be `minmax(0, 1fr)`, not `1fr`: a bare `1fr` has an automatic minimum, so the conversation would grow the row again.
- `max-height`, together with the sidebar's own `overflow-y: auto`, keeps the controls scrollable.
- Test at 375 px wide in DevTools' device toolbar.

### Add styling for a new message role

`messageEl()` sets `class="msg <role>"` (`app.js:127`) and picks the avatar label: `You` for user, `!` for error, `◈` for anything else (128). For a hypothetical `tool` role:

```css
.msg.tool .avatar { background: var(--bg-elevated); color: var(--warn); border: 1px solid var(--border); }
.msg.tool .bubble { color: var(--text-dim); }
```

Then give the role its own label in `messageEl()`. `renderAll()` renders any role other than user and assistant through `renderMarkdown()` (`app.js:151-153`), and `send()` never sends those messages to the server (226).

### Change markdown rendering appearance

`renderMarkdown()` (`app.js:53-98`) emits only these elements:
- `p` (with `br`), `h1`–`h3`, and `ul` / `ol` with `li`;
- `blockquote`, one per quoted line;
- inline `code`, and `pre` holding a `.copy-btn` button and a `code.lang-<name>`;
- `strong`, `em`, and `a`, which opens in a new tab.

Style them only through `.bubble …` selectors, so the sidebar controls are untouched. `em` and the `lang-*` classes have no rules: `em` gets browser italics, and there is no syntax highlighting.

A new element needs a renderer rule first. `docs\howto\ui\app.js.md`, "Add a markdown rule (worked example: horizontal rules)", adds `<hr>`; its style is:

```css
.bubble hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
```

### Make keyboard focus visible

Today:
- Line 97 removes the outline from range inputs, and line 114 from every focused `input`. The sliders and the tuned-settings checkbox therefore show no focus at all: the border-colour change on line 114 does nothing on them.
- Fields (the system prompt, the model id, the backend list) and the composer signal focus with an `--accent-dim` border at 1.6–2.2:1, and the Think pill's ring is `--accent-dim` too (2.09:1).
- Buttons keep the browser's ring.

To meet the 3:1 bar for focus indicators:

1. Add `input[type="range"]:focus-visible, .check input:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }`. Its specificity beats `input:focus` on line 114.
2. Change `--accent-dim` to `--accent` in the focus rules on lines 114 (fields), 215 (composer) and 231 (pill). `--accent` is 6.70:1 on `--bg-input` and 7.64:1 on `--bg-panel`.

### Make the small grey text readable

`--text-faint` (`#5c6674`) colours the small print, mostly at 10–12 px: panel titles, fact keys, slider notes, hints, the stats line, the copy button and the unpressed Think pill. Its contrast is 2.97–3.34:1, below the 4.5:1 that WCAG AA asks for. `--text-faint: #7a8494;` reaches 5.15:1 on `--bg`, 4.88:1 on `--bg-panel` and 4.57:1 on `--bg-elevated`, and stays visibly dimmer than `--text-dim` (`#8b95a5`).

### Change the thinking panel's height

`.thinking-body { max-height: 280px; }` (206). Beyond that height the box scrolls. `app.js` keeps it pinned to the newest line only while the reader is within 40 px of its bottom (`isNearBottom(thinkBody, 40)`, `app.js:273`). With `max-height: none` the panel grows with the thought and the conversation scrolls along with it. The panel still collapses to its label when the first answer token arrives.

### Fit the composer on phones

In Safari on iOS and Chrome on Android, `vh` is the large viewport, measured with the toolbars retracted. So with `height: 100vh` (40), the bottom of the page, the composer, falls below the visible area while the address bar is showing. On line 40, follow `height: 100vh;` with `height: 100dvh;`. Browsers that know the dynamic unit use it, and older ones keep `100vh`.

### Respect reduced motion

Three animations ignore the user's motion preference: the loading dot (`pulse`, 73), the live thinking label (`pulse`, 205) and the streaming cursor (`blink`, 183).

```css
@media (prefers-reduced-motion: reduce) {
  .dot.loading, .thinking.live .think-label, .cursor { animation: none; }
}
```

### Show the copy button to keyboard and touch users

```css
.bubble pre:focus-within .copy-btn { opacity: 1; }
@media (hover: none) { .copy-btn { opacity: 1; } }
```

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| The page renders as a browser-default, black-on-white document | `styles.css` was served as `text/plain`. Python's `mimetypes` reads the Windows registry, where some installers map `.css` that way, and in standards mode browsers ignore such a stylesheet. Chromium's console says "Refused to apply style from …". A 404 on `/static/styles.css` looks the same. | Fixed in v0.2 by the pin at `app.py:46-47`; `test_ui_assets_are_served_with_browser_safe_types` (`tests\test_server.py:186`) guards it. |
| CSS edits do not show after a refresh | A cached copy was reused. Static files carry `ETag` and `Last-Modified` but no `Cache-Control`, so freshness is heuristic. | Ctrl+F5, or DevTools → Network → Disable cache |
| The composer slides below the window as the conversation grows | The `#app` row lost `minmax(0, 1fr)`: for example line 40 was replaced wholesale, or a media query sets `auto 1fr` | Keep `minmax(0, 1fr)`, and `auto minmax(0, 1fr)` when stacking. |
| A long code line widens the conversation column instead of scrolling | `min-width: 0` was removed from `#main` (137) or `.bubble` (156) | Restore it. |
| Send and Stop both show all the time | `!important` was removed from `.hidden` (233); `#send-btn, #stop-btn { display: grid; }` (221) outranks a class | Keep `!important`. |
| Focus vanishes when tabbing onto the sliders or the tuned-settings checkbox | `outline: none` on ranges (97) and on every `input:focus` (114) | See "Make keyboard focus visible". |
| Focus on the fields, the composer or the Think pill is hard to see | `--accent-dim` is 1.6–2.2:1 against its surroundings | Use `--accent` for focus indicators. |
| Panel titles, hints and the stats line are hard to read | `--text-faint` is 2.97–3.34:1, mostly at 10–12 px | `--text-faint: #7a8494;` |
| The copy button is invisible when tabbed to, and missing on touch screens | `opacity: 0` until `.bubble pre:hover` (177–180) | See "Show the copy button to keyboard and touch users". |
| A theme change misses some elements, or the pressed Think pill stays blue | A rule was edited instead of a token, or the colour is one of the six hard-coded ones (the pill tint is at 230) | See "Change the theme colours". |
| Scrollbars are default in Firefox | Only `::-webkit-scrollbar` (235–238) is styled, and Firefox does not support it | Add `scrollbar-color: var(--border) transparent;` to `:root`; the property is inherited. The trade-off: Chromium 121+ then ignores the `::-webkit-scrollbar` rules in favour of the standard property, so its scrollbars lose the 9 px width. |
| On a phone the composer hides under the browser toolbar | `height: 100vh` (40) is the large viewport | See "Fit the composer on phones". |
| The sidebar disappears | The window is ≤ 860 px wide (240–243) | Widen it, or see "Make narrow screens usable". |
| Tables in replies show as plain text with pipes | `renderMarkdown()` has no table rule, so the `.bubble table` rules (174–175) never apply | Add a table rule to the renderer first. |
| A new markdown element looks unstyled | `renderMarkdown()` emits a tag with no `.bubble` rule | Add a `.bubble <tag>` rule. |
| Long thoughts are cut to a small box | `.thinking-body { max-height: 280px; }` (206) | See "Change the thinking panel's height". |
| The message box has a resize grip, and a dragged size snaps back on the next keystroke | `resize: vertical` from the shared field rule (112) reaches `#input`, and the auto-height handler (`app.js:547-550`) rewrites the height on every input | Add `resize: none;` to `#input`. |
| After raising the message box's height limit, it still stops at 200 px | Only one of the two caps changed: `max-height: 200px` here (219) and `Math.min(input.scrollHeight, 200)` in `app.js:549` | Change both. |
| A long model id overflows the sidebar, which then scrolls sideways | The `.kv dd` ellipsis rules (80–81) were removed | Restore `overflow: hidden; text-overflow: ellipsis; white-space: nowrap;`. |
| Slider thumbs differ across browsers | WebKit and Firefox use different thumb pseudo-elements (98–101) | Change both rules together. |

## Related files

- `ui\index.html` — the static elements and classes styled here; see `docs\howto\ui\index.html.md`
- `ui\app.js` — emits the message, markdown and thinking markup and toggles every state hook; see `docs\howto\ui\app.js.md`
- `src\kumaru\server\app.py` — serves this file at `/static/styles.css` (226–227) and pins its `text/css` type (43–47)
- `tests\test_server.py` — `test_ui_assets_are_served_with_browser_safe_types` (183) checks that content type
