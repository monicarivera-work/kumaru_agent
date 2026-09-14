# `ui/app.js`

> Dependency-free browser client for chat streaming, rendering, session state, theme, and controls.

**Read this when:** debugging UI behavior or changing browser/API interactions.

---

## What it does
It manages per-tab session ids, theme persistence, safe markdown rendering, health polling, streamed chat over `fetch`, abort handling, clear/history restore, suggestions, and textarea autogrow.

## Why it exists
Kumaru avoids npm, frameworks, and a separate frontend build. The client must safely render untrusted model output while using standard browser APIs only.

## Mental model
Streaming uses `fetch` plus `ReadableStream` because `EventSource` cannot POST JSON. Model output is escaped before limited markdown tags are reintroduced.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `el` | `const el = (id) => document.getElementById(id)` | DOM lookup helper. |
| `SESSION_KEY` | `"kumaru.session"` | Local storage key for tab session. |
| `THEME_KEY` | `"kumaru.theme"` | Local storage key for theme. |
| `sessionId` | `const` | Persistent browser session id. |
| `applyTheme` | `applyTheme(theme)` | Set theme and persist it. |
| `escapeHtml` | `escapeHtml(text)` | Escape untrusted text. |
| `renderMarkdown` | `renderMarkdown(raw)` | Render minimal safe markdown. |
| `addMessage` | `addMessage(role, content = "")` | Add a transcript bubble. |
| `scrollToBottom` | `scrollToBottom()` | Scroll transcript to latest message. |
| `setBusy` | `setBusy(busy)` | Disable/enable input and stop button. |
| `refreshHealth` | `async refreshHealth()` | Update backend status pill. |
| `send` | `async send(message)` | POST streaming chat and render deltas. |
| `autoGrow` | `autoGrow()` | Resize textarea up to 200px. |
| `restoreHistory` | `async restoreHistory()` | Replay server-side session history. |

## How to use it
```javascript
await fetch("/api/health").then((r) => r.json());
```

```javascript
document.querySelector(".suggestion").click();
```

```javascript
const html = renderMarkdown("**bold** and `code`");
```

```javascript
await fetch(`/api/history/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
```

## How to extend it
When adding a new server event field, update the parsing block inside `send()`. Example: if SSE sends `{notice: "..."}`, handle `event.notice` before `event.done` and render it into `hintEl`.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Stop button does not stop server work immediately | Abort closes the browser request; backend may finish cleanup later. | Expect UI cancellation, and check server backend behavior separately. |
| Markdown HTML appears escaped | Renderer escapes first by design. | Add only safe, fixed transformations after `escapeHtml`. |
| History not restored | Session id changed or `/api/history` failed. | Check local storage and server logs. |
| Streaming frames ignored | Frame is malformed or lacks `data:` prefix. | Emit SSE frames as `data: {json}

`. |

## Related files
- [HTML shell](./index.html.md)
- [Styles](./styles.css.md)
- [Chat routes](../src/kumaru/server/routes_chat.md)
