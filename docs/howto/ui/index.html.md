# `ui/index.html`

> Static HTML shell for the browser chat UI.

**Read this when:** changing visible UI structure or adding elements used by JavaScript/CSS.

---

## What it does
It defines the app layout: header, status pill, theme/clear actions, message area with suggestions, composer textarea, send/stop buttons, and metadata line. It loads `styles.css` and `app.js` directly.

## Why it exists
Kumaru intentionally has no frontend build step. A static HTML shell served by FastAPI keeps browser and API on the same origin.

## Mental model
The IDs are the contract with `ui/app.js`; classes are the contract with `ui/styles.css`. Keep accessible roles and labels when changing interactive elements.

## Public API
| Element/ID | Signature | What it's for |
|---|---|---|
| `html[data-theme]` | `data-theme="dark"` | Theme state toggled by JS and styled by CSS. |
| `#status`, `#status-dot`, `#status-text` | status elements | Backend readiness display. |
| `#theme-btn` | button | Toggle light/dark theme. |
| `#clear-btn` | button | Clear current session. |
| `#messages` | main region | Chat transcript container. |
| `#empty-state` | div | Initial help and suggestions. |
| `.suggestion` | buttons | Prefill/send sample prompts. |
| `#chat-form` | form | Composer submission boundary. |
| `#input` | textarea | User message input. |
| `#send-btn` | button | Submit current message. |
| `#stop-btn` | button | Abort streaming generation. |
| `#usage`, `#hint` | spans | Token/time and operator hints. |

## How to use it
```bash
kumaru serve -c configs/default.yaml
# then open http://127.0.0.1:8000
```

```javascript
document.getElementById("input").value = "Hello Kumaru";
document.getElementById("chat-form").requestSubmit();
```

```javascript
document.documentElement.dataset.theme = "light"
```

## How to extend it
Add new controls with stable IDs, then wire them in `app.js` and style them in `styles.css`. Example: add `<button id="export-btn">Export</button>`, then bind a click handler next to the existing `clear-btn` wiring.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Button does nothing | New element ID is not queried or wired in `app.js`. | Add a DOM lookup and event listener. |
| Element unstyled | Class does not match CSS selectors. | Reuse existing classes or add styles. |
| API calls fail due to origin | Page is not served by Kumaru app. | Open the UI through `kumaru serve`, not a separate file origin. |

## Related files
- [Styles](./styles.css.md)
- [Browser client](./app.js.md)
- [Server app](../src/kumaru/server/app.md)
