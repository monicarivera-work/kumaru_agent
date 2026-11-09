// The connection pill in the header and the dialog that sets the in-memory API key.
import { api, hasApiKey, setApiKey } from "./api.js";
import { openDialog } from "./dialogs.js";
import { el } from "./dom.js";
import { textField } from "./forms.js";
import { emit, on, state } from "./state.js";

const STATES = {
  connecting: { text: "Connecting…", tone: "dim" },
  ok: { text: "Connected", tone: "ok" },
  key: { text: "API key needed", tone: "warn" },
  offline: { text: "Server unreachable", tone: "err" },
};

export function setConnection(next) {
  if (state.connection === next) return;
  state.connection = next;
  emit("connection-changed");
}

export function initConnectionPill(pill) {
  const render = () => {
    const info = STATES[state.connection];
    pill.className = `status-pill tone-${info.tone}`;
    pill.replaceChildren(el("span", { class: "pill-dot", "aria-hidden": "true" }), el("span", {}, info.text));
    pill.setAttribute("aria-label", `${info.text}. Open connection settings`);
  };
  pill.addEventListener("click", (event) => openConnectionDialog(event.currentTarget));
  on("connection-changed", render);
  render();
}

function connectionHint() {
  if (state.connection === "key") return hasApiKey() ? "The current key was rejected. Enter the correct one." : "This server requires an API key.";
  if (state.connection === "ok" && hasApiKey()) return "A key is already set. Enter a new one to replace it.";
  return "Leave empty if the server does not require one.";
}

export async function openConnectionDialog(trigger) {
  const connected = await openDialog({
    title: "Connect to Kumaru", trigger, size: "sm", submitLabel: "Connect",
    description: "Needed only when the server has an API key configured. The key stays in this page's memory: it is not saved to browser storage, cookies or the address bar, and reloading the page forgets it.",
    build: () => textField({
      name: "key", label: "API key", type: "password", required: state.connection === "key",
      hint: connectionHint(),
    }).node,
    onSubmit: async (data) => {
      const previous = setApiKey(String(data.get("key")));
      try {
        await api("/overview");
      } catch (error) {
        setApiKey(previous);
        throw error;
      }
      return true;
    },
  });
  if (connected) emit("connect-requested");
}
