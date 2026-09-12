/* Kumaru chat UI.
 *
 * Three things worth understanding before you edit this file:
 *
 * 1. THE CLIENT OWNS THE CONVERSATION. Every request carries the full message
 *    history, exactly like the OpenAI API. The server stores nothing, which is
 *    why you can run many server replicas behind a load balancer with no
 *    session affinity. Persistence lives in localStorage, here.
 *
 * 2. STREAMING USES fetch + ReadableStream, NOT EventSource. EventSource only
 *    does GET, and we need to POST a conversation. So we read the response body
 *    as a stream and parse the SSE frames by hand — about 20 lines, below.
 *
 * 3. MARKDOWN IS RENDERED ESCAPE-FIRST. All HTML is escaped before any markup
 *    is generated, so model output can never inject script tags. A model is an
 *    untrusted input source; treat its output exactly like user input.
 */

const $ = (id) => document.getElementById(id);

const state = {
  messages: JSON.parse(localStorage.getItem("kumaru.messages") || "[]"),
  streaming: false,
  controller: null,
  info: null,
};

/* ─────────────────────────── markdown ─────────────────────────── */

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function renderMarkdown(src) {
  // Escape everything up front. Nothing below can reintroduce raw HTML.
  let text = escapeHtml(src);

  // Pull fenced code blocks out first so inline rules never rewrite code.
  const blocks = [];
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, body) => {
    blocks.push(
      `<pre><button class="copy-btn" data-copy>copy</button><code class="lang-${lang}">${body.replace(/\n$/, "")}</code></pre>`
    );
    return `\u0000BLOCK${blocks.length - 1}\u0000`;
  });

  text = text
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // Group consecutive list items into a single <ul>/<ol>.
  text = text.replace(/(?:^[-*] .+(?:\n|$))+/gm, (m) => {
    const items = m.trim().split("\n").map((l) => `<li>${l.replace(/^[-*] /, "")}</li>`).join("");
    return `<ul>${items}</ul>\n`;
  });
  text = text.replace(/(?:^\d+\. .+(?:\n|$))+/gm, (m) => {
    const items = m.trim().split("\n").map((l) => `<li>${l.replace(/^\d+\. /, "")}</li>`).join("");
    return `<ol>${items}</ol>\n`;
  });

  // Paragraphs: split on blank lines, leave block-level markup alone.
  text = text
    .split(/\n{2,}/)
    .map((p) => {
      const t = p.trim();
      if (!t) return "";
      if (/^<(h[1-3]|ul|ol|pre|blockquote|table)/.test(t) || t.startsWith("\u0000BLOCK")) return t;
      return `<p>${t.replace(/\n/g, "<br>")}</p>`;
    })
    .join("\n");

  return text.replace(/\u0000BLOCK(\d+)\u0000/g, (_, i) => blocks[+i]);
}

/* ─────────────────────────── rendering ─────────────────────────── */

function messageEl(role, html, meta) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const label = role === "user" ? "You" : role === "error" ? "!" : "◈";
  wrap.innerHTML =
    `<div class="avatar">${label}</div>` +
    `<div class="bubble">${html}${meta ? `<div class="msg-meta">${meta}</div>` : ""}</div>`;
  return wrap;
}

function renderAll() {
  const box = $("messages");
  box.innerHTML = "";
  if (!state.messages.length) {
    box.innerHTML = `<div class="empty-state">
      <h2>Ask anything</h2>
      <p>Running entirely on your machine. Nothing leaves this computer.</p>
      <div class="suggestions">
        <button class="chip">Explain what a KV cache is and why it matters</button>
        <button class="chip">Write a Python function to parse an ISO 8601 duration</button>
        <button class="chip">What's the difference between LoRA and full fine-tuning?</button>
      </div></div>`;
    wireChips();
    return;
  }
  for (const m of state.messages) {
    const html = m.role === "user" ? `<p>${escapeHtml(m.content).replace(/\n/g, "<br>")}</p>` : renderMarkdown(m.content);
    box.appendChild(messageEl(m.role, html, m.meta));
  }
  wireCopy();
  scrollToBottom();
}

function scrollToBottom() {
  const box = $("messages");
  box.scrollTop = box.scrollHeight;
}

function persist() {
  localStorage.setItem("kumaru.messages", JSON.stringify(state.messages.slice(-100)));
}

function wireChips() {
  document.querySelectorAll(".chip").forEach((c) =>
    c.addEventListener("click", () => {
      $("input").value = c.textContent;
      send();
    })
  );
}

function wireCopy() {
  document.querySelectorAll("[data-copy]").forEach((btn) =>
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(btn.parentElement.querySelector("code").textContent);
      btn.textContent = "copied";
      setTimeout(() => (btn.textContent = "copy"), 1200);
    })
  );
}

/* ─────────────────────────── streaming ─────────────────────────── */

async function send() {
  const input = $("input");
  const content = input.value.trim();
  if (!content || state.streaming) return;

  input.value = "";
  input.style.height = "auto";
  state.messages.push({ role: "user", content });
  renderAll();
  persist();

  // Outbound payload: the server is stateless, so send everything.
  const payload = [];
  const sys = $("system-prompt").value.trim();
  if (sys) payload.push({ role: "system", content: sys });
  for (const m of state.messages) {
    if (m.role === "user" || m.role === "assistant") payload.push({ role: m.role, content: m.content });
  }

  setStreaming(true);
  state.controller = new AbortController();

  const node = messageEl("assistant", '<span class="cursor"></span>');
  $("messages").appendChild(node);
  const bubble = node.querySelector(".bubble");
  scrollToBottom();

  let acc = "";
  let meta = "";
  let pending = false;

  // Re-render at most once per animation frame. Without this, a fast GPU
  // triggers hundreds of layout passes per second and the UI stutters even
  // though the model is keeping up.
  const paint = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      bubble.innerHTML = renderMarkdown(acc) + '<span class="cursor"></span>';
      scrollToBottom();
    });
  };

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: payload,
        stream: true,
        temperature: parseFloat($("temperature").value),
        max_tokens: parseInt($("max-tokens").value, 10),
        top_p: parseFloat($("top-p").value),
      }),
      signal: state.controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line; the last fragment may be
      // partial, so keep it in the buffer for the next read.
      const frames = buffer.split("\n\n");
      buffer = frames.pop();

      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;

        let evt;
        try { evt = JSON.parse(data); } catch { continue; }
        if (evt.error) throw new Error(evt.error.message);

        const delta = evt.choices?.[0]?.delta?.content;
        if (delta) { acc += delta; paint(); }

        if (evt.kumaru) {
          const k = evt.kumaru;
          updateStats(k);
          meta = `${k.tokens_per_second ?? "?"} tok/s · first token ${Math.round(k.time_to_first_token_ms ?? 0)}ms · ${k.completion_tokens ?? 0} tokens`;
        }
      }
    }

    state.messages.push({ role: "assistant", content: acc || "(empty response)", meta });
  } catch (e) {
    if (e.name === "AbortError") {
      if (acc) state.messages.push({ role: "assistant", content: acc, meta: "stopped" });
    } else {
      state.messages.push({ role: "error", content: `**Request failed:** ${e.message}` });
    }
  } finally {
    setStreaming(false);
    state.controller = null;
    persist();
    renderAll();
  }
}

function setStreaming(on) {
  state.streaming = on;
  $("send-btn").classList.toggle("hidden", on);
  $("stop-btn").classList.toggle("hidden", !on);
  $("input").disabled = on;
}

function updateStats(k) {
  if (k.time_to_first_token_ms != null) $("stat-ttft").textContent = `${Math.round(k.time_to_first_token_ms)} ms`;
  if (k.tokens_per_second != null) $("stat-tps").textContent = `${k.tokens_per_second} tok/s`;
  if (k.completion_tokens != null) $("stat-tokens").textContent = `${k.prompt_tokens ?? 0} → ${k.completion_tokens}`;
}

/* ─────────────────────────── status ─────────────────────────── */

async function refreshInfo() {
  try {
    const res = await fetch("/v1/info");
    const info = await res.json();
    state.info = info;

    const dot = $("status-dot");
    const txt = $("status-text");
    if (info.error) {
      dot.className = "dot err";
      txt.textContent = "load failed";
      $("hint").textContent = info.error;
    } else if (info.ready) {
      dot.className = "dot ok";
      txt.textContent = "ready";
      $("hint").textContent = "Local model · your data stays on device";
    } else {
      dot.className = "dot loading";
      txt.textContent = "loading model…";
      $("hint").textContent = "First run downloads weights — this can take several minutes.";
    }

    const b = info.agent?.backend ?? {};
    const c = info.config?.backend ?? {};
    $("fact-model").textContent = b.model_id || c.model || "—";
    $("fact-model").title = b.model_id || c.model || "";
    $("fact-backend").textContent = b.name || c.name || "—";
    $("fact-device").textContent = b.device || c.device || "—";
    $("fact-dtype").textContent = b.dtype || c.dtype || "—";
    $("fact-context").textContent = b.context_length ? `${b.context_length.toLocaleString()}` : "—";
    $("version").textContent = `v${info.version}`;

    const sel = $("swap-backend");
    if (sel.options.length === 0 && info.available_backends) {
      for (const name of info.available_backends) {
        const opt = document.createElement("option");
        opt.value = opt.textContent = name;
        if (name === (c.name || "")) opt.selected = true;
        sel.appendChild(opt);
      }
      $("swap-model").value = c.model || "";
    }
  } catch {
    $("status-dot").className = "dot err";
    $("status-text").textContent = "server unreachable";
  }
}

/* ─────────────────────────── wiring ─────────────────────────── */

function init() {
  const input = $("input");

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });

  $("send-btn").addEventListener("click", send);
  $("stop-btn").addEventListener("click", () => state.controller?.abort());

  $("clear-btn").addEventListener("click", () => {
    state.messages = [];
    persist();
    renderAll();
  });

  const bind = (id, out, fmt) => {
    const el = $(id);
    const sync = () => ($(out).textContent = fmt(el.value));
    el.addEventListener("input", sync);
    sync();
  };
  bind("temperature", "val-temp", (v) => parseFloat(v).toFixed(2));
  bind("max-tokens", "val-max", (v) => v);
  bind("top-p", "val-topp", (v) => parseFloat(v).toFixed(2));

  $("swap-btn").addEventListener("click", async () => {
    const btn = $("swap-btn");
    btn.disabled = true;
    btn.textContent = "loading…";
    try {
      const res = await fetch("/v1/backend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: $("swap-backend").value, model: $("swap-model").value.trim() }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error?.message || `HTTP ${res.status}`);
      }
    } catch (e) {
      alert(`Could not load model: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Load model";
      refreshInfo();
    }
  });

  renderAll();
  refreshInfo();
  setInterval(refreshInfo, 4000);
  input.focus();
}

document.addEventListener("DOMContentLoaded", init);
