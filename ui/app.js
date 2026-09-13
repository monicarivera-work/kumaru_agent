/*
 * Kumaru browser client.
 *
 * Deliberately dependency-free: no framework, no bundler, no npm install.
 * Open http://127.0.0.1:8000 after `kumaru serve` and this file is the whole
 * front end.
 *
 * Three things worth knowing:
 *
 * 1. Streaming uses `fetch` + a ReadableStream reader rather than the built-in
 *    `EventSource`, because EventSource cannot send a POST body. The framing is
 *    still plain SSE ("data: {...}\n\n"), so the server side stays standard.
 * 2. Model output is never inserted as HTML. `renderMarkdown` escapes
 *    everything first and only then re-introduces a tiny, fixed set of tags
 *    (code blocks, inline code, bold, italic, links). A model that emits
 *    "<img onerror=...>" therefore renders as text, not as an attack.
 * 3. `AbortController` backs the Stop button, so a runaway generation can be
 *    cut off without reloading the page.
 */

const el = (id) => document.getElementById(id);

const messagesEl = el("messages");
const emptyState = el("empty-state");
const formEl = el("chat-form");
const inputEl = el("input");
const sendBtn = el("send-btn");
const stopBtn = el("stop-btn");
const clearBtn = el("clear-btn");
const themeBtn = el("theme-btn");
const usageEl = el("usage");
const hintEl = el("hint");
const statusDot = el("status-dot");
const statusText = el("status-text");

// One session id per browser tab, kept across reloads so history survives F5.
const SESSION_KEY = "kumaru.session";
const THEME_KEY = "kumaru.theme";

const sessionId =
  localStorage.getItem(SESSION_KEY) ||
  (() => {
    const id =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `s-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(SESSION_KEY, id);
    return id;
  })();

let controller = null;

/* ---------------------------------------------------------------- theme -- */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
}

applyTheme(localStorage.getItem(THEME_KEY) || "dark");

themeBtn.addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

/* ------------------------------------------------------------- rendering -- */

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Minimal, escape-first Markdown. Input is untrusted model output. */
function renderMarkdown(raw) {
  const blocks = [];
  // Pull fenced code out first so its contents are never treated as markup.
  let text = escapeHtml(raw).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const cls = lang ? ` class="language-${lang.replace(/[^\w-]/g, "")}"` : "";
    blocks.push(`<pre><code${cls}>${code.replace(/\n$/, "")}</code></pre>`);
    return `\u0000BLOCK${blocks.length - 1}\u0000`;
  });

  text = text
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    // Only http(s) links become anchors; javascript: URLs stay literal text.
    .replace(
      /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );

  const html = text
    .split(/\n{2,}/)
    .map((para) => (para.trim() ? `<p>${para.replace(/\n/g, "<br />")}</p>` : ""))
    .join("");

  return html.replace(/<p>\u0000BLOCK(\d+)\u0000<\/p>|\u0000BLOCK(\d+)\u0000/g, (_, a, b) =>
    blocks[a ?? b]
  );
}

function addMessage(role, content = "") {
  emptyState?.classList.add("hidden");
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "user") {
    bubble.textContent = content; // user text is never interpreted
  } else {
    bubble.innerHTML = renderMarkdown(content);
  }
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return bubble;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  inputEl.disabled = busy;
  stopBtn.classList.toggle("hidden", !busy);
  if (!busy) inputEl.focus();
}

/* ---------------------------------------------------------------- health -- */

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const ok = response.ok && data.ready;
    statusDot.className = `dot ${ok ? "ok" : "bad"}`;
    statusText.textContent = `${data.backend}${data.model ? " · " + data.model : ""}`;
    if (!ok) hintEl.textContent = "backend not ready - check the server logs";
  } catch {
    statusDot.className = "dot bad";
    statusText.textContent = "server unreachable";
  }
}

/* ------------------------------------------------------------ streaming -- */

async function send(message) {
  addMessage("user", message);
  const bubble = addMessage("assistant", "");
  bubble.classList.add("cursor");
  usageEl.textContent = "";
  hintEl.textContent = "";
  setBusy(true);

  controller = new AbortController();
  let answer = "";
  const started = performance.now();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.message || `server returned ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, split).trim();
        buffer = buffer.slice(split + 2);
        if (!frame.startsWith("data:")) continue;

        let event;
        try {
          event = JSON.parse(frame.slice(5).trim());
        } catch {
          continue; // ignore a partial or malformed frame
        }

        if (event.error) throw new Error(event.message || event.error);
        if (event.delta) {
          answer += event.delta;
          bubble.innerHTML = renderMarkdown(answer);
          scrollToBottom();
        }
        if (event.done) {
          const seconds = (performance.now() - started) / 1000;
          const tokens = event.usage?.total_tokens || 0;
          usageEl.textContent = tokens
            ? `${tokens} tokens · ${seconds.toFixed(1)}s`
            : `${seconds.toFixed(1)}s`;
        }
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      hintEl.textContent = "stopped";
    } else {
      bubble.classList.add("error");
      bubble.textContent = `Error: ${error.message}`;
    }
  } finally {
    bubble.classList.remove("cursor");
    controller = null;
    setBusy(false);
  }
}

/* ----------------------------------------------------------------- wiring -- */

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message || sendBtn.disabled) return;
  inputEl.value = "";
  autoGrow();
  send(message);
});

// Enter sends, Shift+Enter inserts a newline.
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 200)}px`;
}

inputEl.addEventListener("input", autoGrow);

stopBtn.addEventListener("click", () => controller?.abort());

clearBtn.addEventListener("click", async () => {
  controller?.abort();
  await fetch(`/api/history/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  messagesEl.querySelectorAll(".msg").forEach((node) => node.remove());
  emptyState?.classList.remove("hidden");
  usageEl.textContent = "";
  hintEl.textContent = "";
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    inputEl.value = button.textContent.trim();
    autoGrow();
    formEl.requestSubmit();
  });
});

/** Replay server-side history so a reload does not lose the conversation. */
async function restoreHistory() {
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(sessionId)}`);
    if (!response.ok) return;
    const data = await response.json();
    data.messages.forEach((message) => addMessage(message.role, message.content));
  } catch {
    /* first run, or the server is not up yet */
  }
}

restoreHistory();
refreshHealth();
setInterval(refreshHealth, 30000);
