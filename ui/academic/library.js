// The Library list: filters in the rail, search and sort, rows with score meters, pagination
// and the honest empty, loading and error states.
import { api, ApiError } from "./api.js";
import { openConnectionDialog } from "./connection.js";
import { applyPendingFocus } from "./dialogs.js";
import {
  ACCESS, ACCESS_ORDER, accessBadge, authorsLine, badge, button, clear, el, emptyState, fmtInt, kindIcon, meter, originBadge,
  skeletonRows, uid,
} from "./dom.js";
import { icon } from "./icons.js";
import { openAddSourceDialog, openDiscoverDialog } from "./library-dialogs.js";
import { emit, gate, navigate, on, sourceHash, state } from "./state.js";
import { breadcrumb, topicName } from "./topics.js";

const PAGE = 30;
const requests = gate();
let offset = 0;
let total = 0;
let loaded = false;
let hosts = {};

/* ─── Filters ─── */

function filterGroup(title, key, options) {
  const labelId = uid("filter");
  return el("div", { class: "filter-group", role: "group", "aria-labelledby": labelId },
    el("h3", { class: "filter-title", id: labelId }, title),
    el("div", { class: "filter-list" }, options.map((option) => {
      const pressed = (state.filters[key] || "") === option.value;
      return el("button", {
        type: "button", class: ["filter", pressed && "is-pressed"], "aria-pressed": pressed,
        onclick: () => {
          state.filters[key] = pressed ? "" : option.value;
          offset = 0;
          renderFilters();
          reload();
        },
      }, el("span", { class: "filter-label" }, option.label), filterCount(option.count));
    })));
}

function filterCount(count) {
  if (count === null) {
    return el("span", { class: "filter-count" }, el("span", { "aria-hidden": "true" }, "–"), el("span", { class: "sr-only" }, "not loaded"));
  }
  return el("span", { class: "filter-count" }, fmtInt(count));
}

export function renderFilters() {
  const overview = state.overview;
  const counts = overview || { total: 0, by_access: {}, by_kind: {}, by_origin: {}, needs_review: 0 };
  const known = (value) => (overview ? value : null);
  const containers = counts.by_kind.sources_of_sources || 0;
  clear(hosts.filters,
    filterGroup("Access", "access", [
      { value: "", label: "All", count: known(counts.total) },
      ...ACCESS_ORDER.map((access) => ({ value: access, label: ACCESS[access].label, count: known(counts.by_access[access] || 0) })),
    ]),
    filterGroup("Type", "kind", [
      { value: "", label: "All", count: known(counts.total) },
      { value: "works", label: "Individual works", count: known(counts.total - containers) },
      { value: "containers", label: "Sources of Sources", count: known(containers) },
    ]),
    filterGroup("Review", "status", [
      { value: "", label: "All", count: known(counts.total) },
      { value: "needs_review", label: "Needs review", count: known(counts.needs_review) },
      { value: "human_verified", label: "Human verified", count: known(counts.by_origin.human_verified || 0) },
      { value: "ai", label: "AI disclosed", count: known(counts.by_origin.ai || 0) },
    ]));
}

const filtersActive = () => Boolean(state.filters.access || state.filters.kind || state.filters.status || state.filters.q || state.filters.containerId);

export function clearFilters() {
  Object.assign(state.filters, { access: "", kind: "", status: "", q: "", containerId: "", containerTitle: "", containerChildren: 0 });
  hosts.search.value = "";
  offset = 0;
  renderFilters();
  reload();
}

/* ─── Rows ─── */

function scoreMeters(item) {
  if (item.kind === "sources_of_sources") return el("span", { class: "row-scores is-unscored" }, "Not scored");
  const score = item.score;
  const topic = score ? topicName(score.topic_id) : "this subject";
  const label = (name, value) => (typeof value === "number" ? `${name} ${Math.round(value)} out of 100 for ${topic}` : `${name} unknown for ${topic}`);
  return el("span", { class: "row-scores" },
    el("span", { class: "mini" }, el("span", { class: "mini-label", "aria-hidden": "true" }, "L"), meter({ value: score?.legitimacy ?? null, label: label("Legitimacy", score?.legitimacy), compact: true })),
    el("span", { class: "mini" }, el("span", { class: "mini-label", "aria-hidden": "true" }, "U"), meter({ value: score?.usefulness ?? null, label: label("Usefulness", score?.usefulness), compact: true })));
}

function row(item) {
  const selected = state.route.sourceId === item.id;
  const meta = [authorsLine(item.authors, item.author_count), item.publisher, item.published_year].filter(Boolean).join(" · ");
  return el("li", { class: "row-item" },
    el("button", {
      type: "button", class: ["row", selected && "is-selected"], "aria-current": selected ? "true" : undefined, dataset: { id: item.id },
      onclick: () => navigate(sourceHash(item.id, state.route.tab)),
    },
    el("span", { class: "row-kind" }, kindIcon(item.kind)),
    el("span", { class: "row-main" },
      el("span", { class: "row-title" }, item.title),
      meta && el("span", { class: "row-meta" }, meta),
      el("span", { class: "row-badges" },
        accessBadge(item.access), originBadge(item.origin),
        item.pending_notices > 0 && badge({ tone: "warn", iconName: "alert", label: "Notice pending" }),
        item.withdrawn && badge({ tone: "err", iconName: "ban", label: "Retracted" }))),
    scoreMeters(item)));
}

export function markSelected() {
  if (!hosts.list) return;
  for (const node of hosts.list.querySelectorAll(".row")) {
    const selected = node.dataset.id === state.route.sourceId;
    node.classList.toggle("is-selected", selected);
    if (selected) node.setAttribute("aria-current", "true"); else node.removeAttribute("aria-current");
  }
}

function renderContext() {
  const { containerId, containerTitle, containerChildren } = state.filters;
  const hint = containerId && containerChildren > total
    ? el("span", { class: "context-hint" }, `${fmtInt(total)} of ${fmtInt(containerChildren)} contained works are tagged in this subject.`) : null;
  clear(hosts.context,
    el("div", { class: "context-line" }, breadcrumb(state.topicId),
      el("span", { class: "context-count", "aria-live": "polite" }, `${fmtInt(total)} ${total === 1 ? "source" : "sources"}`)),
    containerId && el("div", { class: "context-chips" },
      el("button", {
        type: "button", class: "tag tag-button", "aria-label": `Remove filter: in ${containerTitle}`,
        onclick: () => { Object.assign(state.filters, { containerId: "", containerTitle: "", containerChildren: 0 }); offset = 0; reload(); },
      }, el("span", {}, `In: ${containerTitle}`), icon("x", { size: 12 })), hint));
}

function renderFooter() {
  if (!total) { clear(hosts.footer); return; }
  const first = offset + 1;
  const last = Math.min(offset + PAGE, total);
  clear(hosts.footer,
    el("span", { class: "pager-range", "aria-live": "polite" }, `${fmtInt(first)}–${fmtInt(last)} of ${fmtInt(total)}`),
    el("div", { class: "pager-buttons" },
      button("Previous", { disabled: offset === 0, onclick: () => { offset = Math.max(0, offset - PAGE); reload({ scrollTop: true }); } }),
      button("Next", { disabled: offset + PAGE >= total, onclick: () => { offset += PAGE; reload({ scrollTop: true }); } })));
}

function renderMessage(error) {
  total = 0;
  renderContext();
  renderFooter();
  const needsKey = error instanceof ApiError && error.status === 401;
  clear(hosts.list, emptyState({
    tone: "error",
    title: needsKey ? "API key needed" : error.unreachable ? "Kumaru is not reachable" : "The library could not be loaded",
    body: error.message,
    actions: needsKey ? [button("Enter API key", { variant: "primary", onclick: (event) => openConnectionDialog(event.currentTarget) })]
      : [button("Try again", { variant: "primary", onclick: () => reload() })],
  }));
  hosts.list.removeAttribute("aria-busy");
}

function renderEmpty() {
  if (filtersActive()) {
    clear(hosts.list, emptyState({
      title: "No sources match",
      body: state.filters.containerId
        ? "Nothing in this container is tagged with the selected subject and filters. Pick another subject, or clear the filters."
        : "Nothing in this subject matches the current search and filters.",
      actions: [button("Clear filters", { variant: "primary", onclick: clearFilters })],
    }));
    return;
  }
  clear(hosts.list, emptyState({
    title: `No sources in ${topicName(state.topicId)} yet`,
    body: "Discovery contacts an external provider and returns unreviewed candidates. You can also add a source you already know; nothing is approved automatically.",
    actions: [
      button("Discover with Crossref", { variant: "primary", iconName: "search", onclick: (event) => openDiscoverDialog(event.currentTarget) }),
      button("Add a known source", { iconName: "plus", onclick: (event) => openAddSourceDialog(event.currentTarget) }),
    ],
  }));
}

export async function reload({ scrollTop = false } = {}) {
  if (state.connection === "connecting") return;
  const token = requests.next();
  if (!loaded) clear(hosts.list, skeletonRows());
  hosts.list.setAttribute("aria-busy", "true");
  const { access, kind, status, q, sort, containerId } = state.filters;
  try {
    const result = await api("/library", {
      params: { topic_id: state.topicId, access, kind, status, q, sort, container_id: containerId, limit: PAGE, offset },
    });
    if (!requests.current(token)) return;
    if (result.items.length === 0 && result.total > 0 && offset > 0) { offset = Math.max(0, (Math.ceil(result.total / PAGE) - 1) * PAGE); reload(); return; }
    loaded = true;
    total = result.total;
    renderContext();
    renderFooter();
    if (!result.items.length) renderEmpty();
    else clear(hosts.list, el("ul", { class: "rows", role: "list" }, result.items.map(row)));
    hosts.list.removeAttribute("aria-busy");
    if (scrollTop) hosts.list.scrollTop = 0;
    applyPendingFocus();
  } catch (error) {
    if (!requests.current(token)) return;
    loaded = false;
    renderMessage(error);
  }
}

export function initLibrary({ list, context, footer, filters, search, sort, discover }) {
  hosts = { list, context, footer, filters, search };
  let timer = 0;
  search.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => { state.filters.q = search.value.trim(); offset = 0; reload(); }, 250);
  });
  sort.addEventListener("change", () => { state.filters.sort = sort.value; offset = 0; reload(); });
  discover.addEventListener("click", (event) => openDiscoverDialog(event.currentTarget));
  list.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    const rows = [...list.querySelectorAll(".row")];
    const index = rows.indexOf(document.activeElement);
    if (index < 0) return;
    event.preventDefault();
    const target = { ArrowDown: index + 1, ArrowUp: index - 1, Home: 0, End: rows.length - 1 }[event.key];
    rows[Math.max(0, Math.min(rows.length - 1, target))].focus();
  });
  on("topic-selected", () => { offset = 0; renderFilters(); reload(); });
  on("overview-changed", renderFilters);
  on("data-changed", () => reload());
  on("filters-changed", () => { offset = 0; renderFilters(); reload(); });
  renderFilters();
  emit("library-ready");
}
