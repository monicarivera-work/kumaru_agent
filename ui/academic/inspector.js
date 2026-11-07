// The Inspector: header, actions and tabs for the selected source, or the library at a glance.
import { api } from "./api.js";
import { checkBlockers, collectBlockers, runSourceAction } from "./actions.js";
import { renderClaimsTab } from "./claims.js";
import { applyPendingFocus, toast } from "./dialogs.js";
import {
  accessBadge, authorsLine, badge, button, clear, copyText, el, emptyState, fmtInt, kindBadge, kindIcon, link, originBadge,
  permissionBadge, skeletonRows,
} from "./dom.js";
import { renderEvidenceTab } from "./evidence.js";
import { historyTimeline } from "./history.js";
import { icon } from "./icons.js";
import { renderNoticesTab } from "./notices.js";
import { renderGlance, renderOverviewTab } from "./overview.js";
import { openRelationshipDialog, renderRelationsTab } from "./relations.js";
import { openReviewDialog } from "./review.js";
import { emit, gate, navigate, on, sourceHash, state, viewHash } from "./state.js";
import { topicName } from "./topics.js";

export const TABS = ["overview", "evidence", "claims", "notices", "relations", "history"];
const LABELS = { overview: "Overview", evidence: "Evidence", claims: "Claims", notices: "Notices", relations: "Relations", history: "History" };
const requests = gate();
let host = null;
let current = null;

function renderHistoryTab(ctx) {
  return el("div", { class: "tab-body" },
    el("div", { class: "tab-head" }, el("h3", { class: "tab-title" }, "Review history")),
    el("p", { class: "quiet" }, "Append-only: a review never overwrites an earlier one. Newest first."),
    historyTimeline(ctx.detail.reviews));
}

const RENDERERS = {
  overview: renderOverviewTab, evidence: renderEvidenceTab, claims: renderClaimsTab, notices: renderNoticesTab,
  relations: renderRelationsTab, history: renderHistoryTab,
};

function context() {
  const { id, detail } = current;
  return {
    id, detail, source: detail.source, summary: detail.review_summary, claimId: state.route.claimId,
    container: detail.source.kind === "sources_of_sources",
    go: (tab, claimId = "") => navigate(sourceHash(id, tab, claimId), { replace: true }),
  };
}

function tabCount(tab, ctx) {
  return { evidence: ctx.source.evidence.length, claims: ctx.summary.claim_count, notices: ctx.summary.notice_count, relations: ctx.detail.graph.links.length }[tab];
}

function header(ctx) {
  const { source, summary } = ctx;
  const authors = source.authors.length > 6 ? `${source.authors.slice(0, 6).join(", ")} and ${fmtInt(source.authors.length - 6)} more` : source.authors.join(", ");
  const counts = [
    source.citation_count !== null && `Cited by ${fmtInt(source.citation_count)}`,
    source.reference_count !== null && `${fmtInt(source.reference_count)} references`,
  ].filter(Boolean).join(" · ");
  const meta = [authors, source.publisher, source.published_year].filter(Boolean).join(" · ");
  const blockers = collectBlockers(source, summary);
  const checkMissing = checkBlockers(source);
  const action = (key, label, options = {}) => button(label, { focusId: `action-${key}`, ...options });
  return el("header", { class: "inspector-head" },
    el("button", { type: "button", class: "btn btn-quiet back-button", onclick: () => navigate(viewHash("library")) }, icon("back"), el("span", {}, "Library")),
    el("div", { class: "title-row" },
      el("span", { class: "title-icon" }, kindIcon(source.kind)),
      el("h2", { id: "inspector-title", class: "inspector-title", tabindex: "-1" }, source.title)),
    el("p", { class: "source-url" }, link(source.url)),
    el("div", { class: "badge-row" },
      kindBadge(source.kind), accessBadge(source.access, { full: true }), originBadge(source.origin), permissionBadge(source.permission),
      source.topic_reviewed ? badge({ tone: "ok", iconName: "check", label: "Topics reviewed" })
        : badge({ tone: "dim", iconName: "question", label: "Topics are candidates", variant: "dashed" }),
      summary.withdrawn ? badge({ tone: "err", iconName: "ban", label: "Retracted" })
        : summary.pending_notices ? badge({ tone: "warn", iconName: "alert", label: "Notice pending" })
          : badge({ tone: "dim", iconName: "info", label: "No notice reported", title: "A quiet record is not proof that the source is current." })),
    (meta || source.doi || counts) && el("p", { class: "source-meta" }, meta,
      source.doi && [meta ? " · " : "", "DOI ", link(`https://doi.org/${source.doi}`, source.doi)],
      counts && el("span", { class: "source-counts" }, `${meta || source.doi ? " · " : ""}${counts} (provider-reported)`)),
    el("div", { class: "id-row" }, el("span", { class: "id-label" }, "Source ID"), el("code", { class: "source-id" }, source.id),
      button("Copy", { variant: "quiet", iconName: "copy", onclick: async () => toast(await copyText(source.id) ? { message: "Source ID copied." } : { kind: "error", message: "Copy failed. Select the ID and copy it manually." }) })),
    el("div", { class: "actions" },
      action("review", "Review source", { variant: "primary", iconName: "check", onclick: (event) => openReviewDialog({ source, trigger: event.currentTarget }) }),
      action("inspect", "Inspect page", { iconName: "search", onclick: (event) => runSourceAction("inspect", source, event.currentTarget) }),
      action("collect", "Collect permitted text", { disabled: blockers.length > 0, describedBy: blockers.length ? "collect-note" : undefined, onclick: (event) => runSourceAction("collect", source, event.currentTarget) }),
      action("check", "Check editorial updates", { disabled: checkMissing.length > 0, describedBy: checkMissing.length ? "check-note" : undefined, onclick: (event) => runSourceAction("check", source, event.currentTarget) }),
      action("link", "Link source…", { iconName: "link", onclick: (event) => openRelationshipDialog({ source, trigger: event.currentTarget }) })),
    blockers.length > 0 && el("p", { class: "action-note", id: "collect-note" }, `Collect permitted text needs ${blockers.join(", ")}.`),
    checkMissing.length > 0 && el("p", { class: "action-note", id: "check-note" }, `Editorial checks need ${checkMissing.join(", ")}.`));
}

function tabStrip(ctx) {
  const tab = state.route.tab;
  const activate = (name) => {
    ctx.go(name);
    requestAnimationFrame(() => document.getElementById(`tab-${name}`)?.focus());
  };
  const buttons = TABS.map((name) => {
    const count = tabCount(name, ctx);
    return el("button", {
      type: "button", role: "tab", id: `tab-${name}`, class: ["tab", tab === name && "is-selected"], "aria-selected": tab === name,
      "aria-controls": "tab-panel", tabindex: tab === name ? 0 : -1, dataset: { tab: name },
      onclick: () => activate(name),
    }, LABELS[name], count !== undefined && el("span", { class: "tab-badge" }, fmtInt(count)));
  });
  const strip = el("div", { class: "tabs", role: "tablist", "aria-label": "Source sections" }, buttons);
  strip.addEventListener("keydown", (event) => {
    const index = TABS.indexOf(tab);
    const next = { ArrowRight: (index + 1) % TABS.length, ArrowLeft: (index + TABS.length - 1) % TABS.length, Home: 0, End: TABS.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    activate(TABS[next]);
  });
  return strip;
}

function renderAll({ resetScroll = false } = {}) {
  const ctx = context();
  const tab = TABS.includes(state.route.tab) ? state.route.tab : "overview";
  state.route.tab = tab;
  document.title = `${ctx.source.title} · Academic review · Kumaru`;
  const panel = el("div", { id: "tab-panel", role: "tabpanel", class: "tab-panel", "aria-labelledby": `tab-${tab}`, tabindex: "0" }, RENDERERS[tab](ctx));
  const scroller = host.parentElement;
  const top = scroller.scrollTop;
  clear(host, header(ctx), tabStrip(ctx), panel);
  scroller.scrollTop = resetScroll ? 0 : top;
  applyPendingFocus();
  emit("inspector-rendered");
}

function renderLoading() {
  clear(host, el("div", { class: "inspector-head" },
    el("button", { type: "button", class: "btn btn-quiet back-button", onclick: () => navigate(viewHash("library")) }, icon("back"), el("span", {}, "Library")),
    el("h2", { id: "inspector-title", class: "inspector-title", tabindex: "-1" }, "Loading source…")), skeletonRows(3));
}

function renderError(error) {
  clear(host, el("div", { class: "inspector-head" },
    el("button", { type: "button", class: "btn btn-quiet back-button", onclick: () => navigate(viewHash("library")) }, icon("back"), el("span", {}, "Library")),
    el("h2", { id: "inspector-title", class: "inspector-title", tabindex: "-1" }, "This source could not be opened")),
  emptyState({
    tone: "error", title: error.status === 400 ? "Not in the catalogue" : "Could not load the source", body: error.message,
    actions: [button("Back to the library", { variant: "primary", onclick: () => navigate(viewHash("library")) }), button("Try again", { onclick: () => reloadSource() })],
  }));
}

export async function showSource(id, { force = false } = {}) {
  if (current && current.id === id && !force) { renderAll({ resetScroll: true }); return; }
  const token = requests.next();
  if (!current || current.id !== id) { current = null; renderLoading(); }
  try {
    const detail = await api(`/sources/${id}`);
    if (!requests.current(token)) return;
    current = { id, detail };
    renderAll({ resetScroll: !force });
  } catch (error) {
    if (!requests.current(token)) return;
    current = null;
    renderError(error);
  }
}

export function showGlance() {
  requests.next();
  current = null;
  document.title = "Academic review · Kumaru";
  clear(host, renderGlance());
  applyPendingFocus();
}

export const reloadSource = () => (state.route.sourceId ? showSource(state.route.sourceId, { force: true }) : Promise.resolve());
export const focusTitle = () => document.getElementById("inspector-title")?.focus({ preventScroll: true });

export function initInspector(element) {
  host = element;
  on("data-changed", () => { if (state.route.sourceId) reloadSource(); });
  on("overview-changed", () => { if (!state.route.sourceId && host.querySelector(".glance")) showGlance(); });
}
