# `ui/styles.css`

> Dependency-free stylesheet for Kumaru's chat UI, layout, and themes.

**Read this when:** adjusting UI appearance, themes, or responsive behavior.

---

## What it does
It defines CSS custom properties for dark/light themes, a three-row app grid, header/status styles, message bubbles, markdown code formatting, composer layout, hidden state, and mobile behavior.

## Why it exists
With no framework or build step, CSS must carry layout and theme behavior directly. Custom properties let the theme button switch colors by changing one `data-theme` attribute.

## Mental model
Theme variables live on `html[data-theme="dark|light"]`. Layout is header, scrollable messages, fixed composer. `.hidden` is the shared visibility utility used by JavaScript.

## Public API
| Selector/token | Signature | What it's for |
|---|---|---|
| `:root` | CSS variables | Shared radius and fonts. |
| `html[data-theme="dark"]` | theme variables | Default dark palette. |
| `html[data-theme="light"]` | theme variables | Light palette. |
| `.app` | grid container | Header/messages/composer layout. |
| `.topbar`, `.brand`, `.status`, `.actions` | header classes | Header and status layout. |
| `.messages`, `.msg`, `.bubble` | transcript classes | Scrollable chat and bubbles. |
| `.msg.user`, `.msg.assistant`, `.msg.error` | role classes | Message-specific styling. |
| `.cursor::after` | pseudo-element | Streaming caret animation. |
| `.empty`, `.suggestions`, `.suggestion` | empty state | Startup prompt suggestions. |
| `.composer`, `#chat-form`, `textarea`, `.meta` | composer selectors | Input area. |
| `.hidden` | utility | Force-hide elements. |
| `@media (max-width: 600px)` | media query | Compact mobile header. |

## How to use it
```javascript
document.documentElement.dataset.theme = "dark";
```

```javascript
document.querySelector("#stop-btn").classList.remove("hidden");
```

```javascript
document.querySelector("#status-dot").className = "dot ok";
```

## How to extend it
Add colors as CSS variables under both theme blocks before using them. Example: define `--warning` in dark and light themes, then use it in `.msg.warning .bubble`.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Theme toggle changes nothing | New color used a literal instead of a CSS variable or missing theme variable. | Define the variable in both theme blocks. |
| Composer scrolls away | `.app` grid or viewport height was changed. | Keep three-row grid with messages as the scroll region. |
| Hidden element still occupies space | Element is hidden another way. | Use `.hidden`, which applies `display: none !important`. |

## Related files
- [HTML shell](./index.html.md)
- [Browser client](./app.js.md)
