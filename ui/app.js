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
 *
 * 4. THOUGHTS ARE SHOWN, NEVER SENT BACK. Reasoning models stream their
 *    thinking in `delta.reasoning_content` and the answer in `delta.content`.
 *    Thoughts are displayed and saved locally, but the history sent to the
 *    server carries answers only: models are trained on conversations whose
 *    earlier turns contain answers, not thoughts (Qwen3-style chat templates
 *    discard old thoughts anyway), and resending them would burn context.
 */

const $ = (id) => document.getElementById(id);

// Slider stops. Token counts span 2 orders of magnitude, so the sliders move
// through hand-picked, roughly logarithmic stops instead of a linear range.
const MAX_TOKEN_STEPS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
// null = unlimited. Kept below the 32k reply cap so an answer still fits after
// the thinking budget is spent.
const BUDGET_STEPS = [256, 512, 1024, 2048, 4096, 8192, 16384, 24576, null];

const state = {
  messages: JSON.parse(localStorage.getItem("kumaru.messages") || "[]"),
  streaming: false,
  controller: null,
  info: null,
  think: false,          // the Think toggle; starts at the server profile's default
  thinkTouched: false,   // once the user clicks it, the server default stops applying
  controlsReady: false,
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

function formatSeconds(ms) {
  const s = ms / 1000;
  if (s < 10) return `${s.toFixed(1)}s`;
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

// Escaped, never parsed: thoughts are plain text shown with `white-space: pre-wrap`.
function thinkingHtml(text, label) {
  return `<details class="thinking"><summary><span class="think-label">${escapeHtml(label)}</span></summary>` +
    `<div class="thinking-body">${escapeHtml(text)}</div></details>`;
}

function assistantHtml(m) {
  let html = "";
  if (m.reasoning) {
    html += thinkingHtml(m.reasoning, m.thinkingMs != null ? `Thought for ${formatSeconds(m.thinkingMs)}` : "Thoughts");
  }
  // An answer can be empty: stopped mid-thought, or cut off by max tokens.
  html += m.content ? renderMarkdown(m.content) : `<p class="note">${escapeHtml(m.note || "(empty response)")}</p>`;
  return html;
}

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
    const html = m.role === "user"
      ? `<p>${escapeHtml(m.content).replace(/\n/g, "<br>")}</p>`
      : m.role === "assistant" ? assistantHtml(m) : renderMarkdown(m.content);
    box.appendChild(messageEl(m.role, html, m.meta));
  }
  wireCopy();
  scrollToBottom();
}

function scrollToBottom() {
  const box = $("messages");
  // "instant": a smooth scroll restarted 60 times a second never catches up,
  // and it would make isNearBottom() lie mid-animation.
  box.scrollTo({ top: box.scrollHeight, behavior: "instant" });
}

// Follow the stream only if the reader is already at the bottom; someone who
// scrolled up to reread something should not be yanked back down.
function isNearBottom(el = $("messages"), slack = 120) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < slack;
}

function persist() {
  const recent = state.messages.slice(-100);
  try {
    localStorage.setItem("kumaru.messages", JSON.stringify(recent));
  } catch {
    // Long thoughts can exceed the ~5 MB localStorage quota. Answers matter
    // more than thoughts, so drop the thoughts and try once more.
    try {
      localStorage.setItem("kumaru.messages", JSON.stringify(recent.map(({ reasoning, ...m }) => m)));
    } catch {
      // Storage full or disabled: chat still works, it just won't survive a reload.
    }
  }
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

  // Outbound payload: the server is stateless, so send everything — answers
  // only. Thoughts, notes, and empty (stopped) replies never go back.
  const payload = [];
  const sys = $("system-prompt").value.trim();
  if (sys) payload.push({ role: "system", content: sys });
  for (const m of state.messages) {
    if ((m.role === "user" || m.role === "assistant") && m.content) {
      payload.push({ role: m.role, content: m.content });
    }
  }

  setStreaming(true);
  state.controller = new AbortController();

  // Built once and then updated in place: rebuilding it every frame would
  // reset the thinking block whenever the user opens or closes it.
  const node = messageEl("assistant", "");
  const bubble = node.querySelector(".bubble");
  bubble.innerHTML =
    '<details class="thinking live hidden" open>' +
    '<summary><span class="think-label">Thinking…</span></summary>' +
    '<div class="thinking-body"></div></details>' +
    '<div class="answer"><span class="cursor"></span></div>';
  const details = bubble.querySelector(".thinking");
  const thinkLabel = bubble.querySelector(".think-label");
  const thinkBody = bubble.querySelector(".thinking-body");
  const thinkText = thinkBody.appendChild(document.createTextNode(""));
  const answer = bubble.querySelector(".answer");
  $("messages").appendChild(node);
  scrollToBottom();

  let acc = "";            // the answer
  let reasoning = "";      // the thoughts
  let painted = 0;         // how much of `reasoning` is already on screen
  let thinkStart = null;   // client-side clock; the server's figure replaces it at the end
  let thinkEnd = null;
  let thinkingMs = null;
  let finish = null;
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
      const follow = isNearBottom();
      if (reasoning.length > painted) {
        // Append only the new text: constant work per frame however long the
        // thought grows, and a text node never parses HTML.
        const followThought = isNearBottom(thinkBody, 40);
        thinkText.appendData(reasoning.slice(painted));
        painted = reasoning.length;
        details.classList.remove("hidden");
        if (followThought) thinkBody.scrollTop = thinkBody.scrollHeight;
      }
      if (thinkStart !== null) {
        thinkLabel.textContent = thinkEnd === null
          ? `Thinking… ${Math.floor((performance.now() - thinkStart) / 1000)}s`
          : `Thought for ${formatSeconds(thinkEnd - thinkStart)}`;
      }
      answer.innerHTML = acc
        ? renderMarkdown(acc) + '<span class="cursor"></span>'
        : reasoning ? "" : '<span class="cursor"></span>';
      if (follow) scrollToBottom();
    });
  };

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: payload, stream: true, ...requestSettings() }),
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

        const choice = evt.choices?.[0];
        const thought = choice?.delta?.reasoning_content;
        if (thought) {
          if (thinkStart === null) thinkStart = performance.now();
          reasoning += thought;
          paint();
        }
        const text = choice?.delta?.content;
        if (text) {
          if (thinkStart !== null && thinkEnd === null) {
            // The answer has started: fold the thoughts away.
            thinkEnd = performance.now();
            details.open = false;
            details.classList.remove("live");
          }
          acc += text;
          paint();
        }
        if (choice?.finish_reason) finish = choice.finish_reason;

        if (evt.kumaru) {
          const k = evt.kumaru;
          updateStats(k);
          if (k.thinking_ms != null) thinkingMs = k.thinking_ms;
          meta = `${k.tokens_per_second ?? "?"} tok/s · first token ${Math.round(k.time_to_first_token_ms ?? 0)}ms · ${k.completion_tokens ?? 0} tokens`;
          if (k.thinking_ms != null) meta += ` · thought ${formatSeconds(k.thinking_ms)}`;
        }
      }
    }

    let note;
    if (finish === "length") {
      meta += " · cut off at max tokens";
      if (!acc) {
        note = reasoning
          ? "No answer: the max-tokens cap ran out while the model was still thinking. Raise max tokens or set a thinking budget."
          : "No answer: the max-tokens cap ran out.";
      }
    }
    state.messages.push(finished(acc, reasoning, thinkingMs, thinkStart, thinkEnd, meta, note));
  } catch (e) {
    if (e.name === "AbortError") {
      if (acc || reasoning) {
        state.messages.push(
          finished(acc, reasoning, thinkingMs, thinkStart, thinkEnd, "stopped", "Stopped before answering.")
        );
      }
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

// The stored form of a finished reply. Thinking time prefers the server's
// measurement (no network jitter) and falls back to the client's clock.
function finished(content, reasoning, serverMs, start, end, meta, note) {
  const m = { role: "assistant", content, meta };
  if (reasoning) {
    m.reasoning = reasoning;
    m.thinkingMs = serverMs ?? ((end ?? performance.now()) - start);
  }
  if (note && !content) m.note = note;
  return m;
}

// What this request asks for beyond the messages. With "tuned settings" on,
// sampling fields are omitted so the server's profile decides; that profile
// has separate presets for thinking and non-thinking replies.
function requestSettings() {
  const s = { thinking: state.think };
  if (state.think) {
    const budget = BUDGET_STEPS[+$("thinking-budget").value];
    if (budget != null) s.thinking_budget = budget;
  }
  if (!$("auto-sampling").checked) {
    s.temperature = parseFloat($("temperature").value);
    s.top_p = parseFloat($("top-p").value);
    s.max_tokens = MAX_TOKEN_STEPS[+$("max-tokens").value];
  }
  return s;
}

function setStreaming(on) {
  state.streaming = on;
  $("send-btn").classList.toggle("hidden", on);
  $("stop-btn").classList.toggle("hidden", !on);
  $("input").disabled = on;
}

function updateStats(k) {
  if (k.time_to_first_token_ms != null) $("stat-ttft").textContent = `${Math.round(k.time_to_first_token_ms)} ms`;
  $("stat-think").textContent = k.thinking_ms != null ? formatSeconds(k.thinking_ms) : "off";
  if (k.tokens_per_second != null) $("stat-tps").textContent = `${k.tokens_per_second} tok/s`;
  if (k.completion_tokens != null) $("stat-tokens").textContent = `${k.prompt_tokens ?? 0} → ${k.completion_tokens}`;
}

/* ─────────────────────────── controls ─────────────────────────── */

function nearestIndex(steps, value) {
  let best = 0;
  steps.forEach((s, i) => {
    if (s != null && Math.abs(Math.log(s / value)) < Math.abs(Math.log(steps[best] / value))) best = i;
  });
  return best;
}

// The values the server will use when the request leaves sampling unset:
// `generation`, overlaid with `thinking_mode` when thinking is on (the same
// precedence the server applies in ChatAgent.resolve_params).
function profileDefaults() {
  const cfg = state.info?.config;
  if (!cfg) return null;
  const eff = { ...cfg.generation };
  if (state.think) {
    for (const [k, v] of Object.entries(cfg.thinking_mode || {})) if (v != null) eff[k] = v;
  }
  return eff;
}

// One function brings every control and label in line with the current state;
// every event handler just calls it.
function syncControls() {
  const auto = $("auto-sampling").checked;
  for (const id of ["temperature", "top-p", "max-tokens"]) {
    $(id).disabled = auto;
    $(id).closest(".slider").classList.toggle("is-disabled", auto);
  }

  const eff = auto ? profileDefaults() : null;
  if (eff) {
    // Park the inert sliders on the profile's values, so unchecking "tuned
    // settings" starts from them rather than jumping somewhere else.
    $("temperature").value = eff.temperature;
    $("top-p").value = eff.top_p;
    $("max-tokens").value = nearestIndex(MAX_TOKEN_STEPS, eff.max_tokens);
  }
  const temp = eff ? eff.temperature : parseFloat($("temperature").value);
  const topP = eff ? eff.top_p : parseFloat($("top-p").value);
  const maxTok = eff ? eff.max_tokens : MAX_TOKEN_STEPS[+$("max-tokens").value];
  $("val-temp").textContent = temp.toFixed(2);
  $("val-topp").textContent = topP.toFixed(2);
  $("val-max").textContent = maxTok.toLocaleString();

  const budget = BUDGET_STEPS[+$("thinking-budget").value];
  $("val-budget").textContent = budget == null ? "unlimited" : budget.toLocaleString();
  $("thinking-budget").disabled = !state.think;
  $("budget-row").classList.toggle("is-disabled", !state.think);

  $("think-btn").setAttribute("aria-pressed", String(state.think));
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

    // The Think toggle starts where the profile says (`generation.thinking`),
    // unless the user has already chosen.
    if (!state.controlsReady && info.config) {
      state.controlsReady = true;
      if (!state.thinkTouched) state.think = info.config.generation?.thinking === true;
    }
    syncControls();
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

  for (const id of ["temperature", "max-tokens", "top-p", "thinking-budget", "auto-sampling"]) {
    $(id).addEventListener("input", syncControls);
  }
  $("think-btn").addEventListener("click", () => {
    state.think = !state.think;
    state.thinkTouched = true;
    syncControls();
  });
  syncControls();

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
