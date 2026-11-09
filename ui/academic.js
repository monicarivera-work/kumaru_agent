// Academic review dashboard: bootstrap, hash router and view switching. Behaviour lives in the
// modules under /static/academic/; this file only wires them to the page shell.
import { api } from "./academic/api.js";
import { initConnectionPill, setConnection } from "./academic/connection.js";
import { closeAllDialogs, confirmDiscard, toast, toastError } from "./academic/dialogs.js";
import { clear, fmtInt, plural, skeletonRows } from "./academic/dom.js";
import { focusTitle, initInspector, showGlance, showSource, TABS } from "./academic/inspector.js";
import { initLibrary, markSelected } from "./academic/library.js";
import { emit, gate, isDirty, navigate, on, state } from "./academic/state.js";
import { loadTopics, renderSubjects } from "./academic/topics.js";
import { initViews, showContainers, showQueue } from "./academic/views.js";

const $ = (id) => document.getElementById(id);
const overviewRequests = gate();
const narrow = window.matchMedia("(max-width: 759px)");
const compact = window.matchMedia("(max-width: 1279px)");
let lastHash = location.hash || "#/library";
let bootstrapped = false;

/* ─── Routing ─── */

function parseRoute() {
  const [view, first, second, third] = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (view === "source" && /^[0-9a-f]{24}$/.test(first || "")) {
    const tab = TABS.includes(second) ? second : "overview";
    return { view: "source", sourceId: first, tab, claimId: tab === "claims" && /^[\w-]+$/.test(third || "") ? third : "" };
  }
  if (view === "containers" || view === "queue") return { view, sourceId: null, tab: "overview", claimId: "" };
  return { view: "library", sourceId: null, tab: "overview", claimId: "" };
}

const TITLES = { library: "Library", containers: "Sources of Sources", queue: "Review queue" };

async function applyRoute({ initial = false } = {}) {
  const previous = state.route;
  const route = parseRoute();
  state.route = route;
  const view = route.view === "source" ? "library" : route.view;
  document.body.dataset.view = view;
  document.body.dataset.screen = route.view === "source" ? "inspector" : "list";
  for (const name of Object.keys(TITLES)) $(`view-${name}`).hidden = name !== view;
  document.querySelectorAll("[data-nav]").forEach((link) => {
    if (link.dataset.nav === view) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  const changedView = (previous.view === "source" ? "library" : previous.view) !== view;
  if (route.view === "source") {
    document.title = "Academic review · Kumaru";
    markSelected();
    await showSource(route.sourceId);
    if (!initial && narrow.matches && previous.sourceId !== route.sourceId) focusTitle();
  } else if (route.view === "library") {
    markSelected();
    showGlance();
    document.title = `${TITLES.library} · Academic review · Kumaru`;
  } else {
    document.title = `${TITLES[view]} · Academic review · Kumaru`;
    if (view === "containers") showContainers(); else showQueue();
  }
  if (!initial && changedView) $(`${view}-title`)?.focus({ preventScroll: true });
}

window.addEventListener("hashchange", async () => {
  const next = location.hash;
  if (isDirty()) {
    if (!(await confirmDiscard())) { history.replaceState(null, "", lastHash); return; }
    closeAllDialogs();
  }
  lastHash = next;
  applyRoute();
});
on("route-replaced", () => { lastHash = location.hash; applyRoute(); });
window.addEventListener("beforeunload", (event) => {
  if (isDirty()) { event.preventDefault(); event.returnValue = ""; }
});

/* ─── Overview, queue badge and monitor ─── */

function renderQueueBadge() {
  const badge = $("queue-count");
  const queue = state.queue;
  const waiting = queue ? queue.needs_review + queue.pending_notices + queue.unreviewed_claims : 0;
  badge.textContent = fmtInt(waiting);
  badge.hidden = waiting === 0;
  badge.setAttribute("aria-label", `${plural(waiting, "item")} awaiting review`);
}

async function refreshOverview() {
  const token = overviewRequests.next();
  const [scoped, global] = await Promise.all([api("/overview", { params: { topic_id: state.topicId } }), api("/overview")]);
  if (!overviewRequests.current(token)) return;
  state.overview = scoped;
  state.queue = global;
  state.formula = scoped.formula;
  state.braveConfigured = scoped.brave_configured;
  state.openalexKeyConfigured = Boolean(scoped.openalex_key_configured);
  state.unpaywallConfigured = Boolean(scoped.unpaywall_configured);
  renderQueueBadge();
  emit("overview-changed");
}

function interval(seconds) {
  if (seconds % 86400 === 0) return `${seconds / 86400} d`;
  if (seconds % 3600 === 0) return `${seconds / 3600} h`;
  return `${Math.max(1, Math.round(seconds / 60))} min`;
}

function renderMonitor() {
  const monitor = state.monitor;
  const text = monitor?.periodic_enabled ? `every ${interval(monitor.interval_s)}` : "manual";
  $("monitor-chip").textContent = `Editorial checks: ${text}`;
  $("monitor-chip").title = monitor?.periodic_enabled
    ? `Batches of ${monitor.batch_size} DOI sources, while the server runs.` : "Nothing is checked unless you ask. A check finding nothing does not prove a source is current.";
}

async function runMonitor(trigger) {
  const label = trigger.querySelector("span");
  const original = label.textContent;
  trigger.disabled = true;
  label.textContent = "Checking…";
  try {
    const result = await api("/monitor", { method: "POST", params: { limit: 10 } });
    const failed = result.errors.length;
    toast({
      kind: failed ? "error" : "success",
      title: "Editorial checks finished",
      message: `${plural(result.results.length, "source")} checked, ${failed} failed. New notices are reports, not confirmed findings.`,
      details: result.errors.map((item) => item.error).join("\n"),
    });
    emit("data-changed", { scope: "monitor" });
  } catch (error) {
    toastError(error, "Editorial checks failed");
  } finally {
    trigger.disabled = false;
    label.textContent = original;
  }
}

/* ─── Drawer and header tools ─── */

function wireDrawer() {
  const drawer = $("subjects");
  const toggle = $("drawer-toggle");
  const scrim = $("drawer-scrim");
  const setOpen = (open, { restore = true } = {}) => {
    if (drawer.classList.contains("is-open") === open) return;
    drawer.classList.toggle("is-open", open);
    scrim.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) drawer.querySelector('[role="treeitem"][tabindex="0"]')?.focus();
    else if (restore) toggle.focus();
  };
  toggle.addEventListener("click", () => setOpen(!drawer.classList.contains("is-open")));
  scrim.addEventListener("click", () => setOpen(false));
  $("drawer-close").addEventListener("click", () => setOpen(false));
  drawer.addEventListener("keydown", (event) => { if (event.key === "Escape" && compact.matches) setOpen(false); });
  drawer.addEventListener("focusout", (event) => {
    if (compact.matches && event.relatedTarget && !drawer.contains(event.relatedTarget) && event.relatedTarget !== toggle) setOpen(false, { restore: false });
  });
  on("topic-selected", () => { if (compact.matches) setOpen(false); });
  compact.addEventListener("change", () => setOpen(false, { restore: false }));
}

function wireTools() {
  const toggle = $("tools-toggle");
  const panel = $("header-tools");
  const setOpen = (open) => {
    panel.classList.toggle("is-open", open);
    toggle.setAttribute("aria-expanded", String(open));
  };
  toggle.addEventListener("click", () => setOpen(!panel.classList.contains("is-open")));
  document.addEventListener("click", (event) => { if (!panel.contains(event.target) && !toggle.contains(event.target)) setOpen(false); });
  panel.addEventListener("keydown", (event) => { if (event.key === "Escape") { setOpen(false); toggle.focus(); } });
  $("monitor-button").addEventListener("click", (event) => runMonitor(event.currentTarget));
  $("skip-link").addEventListener("click", (event) => {
    event.preventDefault();
    const target = state.route.view === "source" || state.route.view === "library" ? $("library-search") : $(`${state.route.view}-title`);
    target?.focus();
  });
}

/* ─── Bootstrap ─── */

async function bootstrap() {
  setConnection("connecting");
  try {
    await loadTopics();
    await refreshOverview();
    state.monitor = await api("/monitor/status");
    renderMonitor();
    setConnection("ok");
  } catch (error) {
    setConnection(error.status === 401 ? "key" : "offline");
  }
  const initial = !bootstrapped;
  bootstrapped = true;
  await applyRoute({ initial });
  emit("filters-changed");
}

on("topic-selected", () => { refreshOverview().catch(toastError); });
on("data-changed", () => { refreshOverview().catch(() => {}); });
on("connect-requested", bootstrap);

initConnectionPill($("connection-pill"));
renderSubjects($("subjects-host"));
initLibrary({
  list: $("library-list"), context: $("library-context"), footer: $("library-footer"), filters: $("filters-host"),
  search: $("library-search"), sort: $("library-sort"), discover: $("discover-button"), freeCopies: $("free-copies-button"),
});
initInspector($("inspector-body"));
initViews({ containers: $("containers-body"), queue: $("queue-body") });
clear($("library-list"), skeletonRows());
wireDrawer();
wireTools();
bootstrap();
