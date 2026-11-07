// Overview tab ("Before relying on this source" and one score card per subject) and the
// "Library at a glance" panel shown when no source is selected.
import { collectBlockers, runSourceAction } from "./actions.js";
import { scoreCard } from "./anatomy.js";
import {
  ACCESS, ACCESS_ORDER, button, el, fmtDate, fmtInt, fmtRelative, plural,
} from "./dom.js";
import { icon } from "./icons.js";
import { openDiscoverDialog } from "./library-dialogs.js";
import { openReviewDialog } from "./review.js";
import { emit, navigate, state, viewHash } from "./state.js";
import { breadcrumb, breadcrumbText, topicName } from "./topics.js";

const ROW_ICONS = { ok: "check", todo: "question", bad: "alert", info: "info" };
const ROW_PREFIX = { ok: "Done. ", todo: "Needs attention. ", bad: "Blocked. ", info: "" };

function checkRow({ label, tone, text, action }) {
  return el("li", { class: `check-row tone-${tone}` },
    el("span", { class: "check-icon" }, icon(ROW_ICONS[tone], { size: 18 })),
    el("div", { class: "check-main" },
      el("strong", { class: "check-label" }, label),
      el("p", { class: "check-status" }, el("span", { class: "sr-only" }, ROW_PREFIX[tone]), text)),
    action && button(action.label, { variant: "secondary", focusId: action.focusId, disabled: action.disabled, onclick: action.onclick }));
}

function checklist(ctx) {
  const { source, summary, detail } = ctx;
  const review = (section) => (event) => openReviewDialog({ source, trigger: event.currentTarget, section });
  const topics = source.topic_ids.map(topicName).join(", ");
  const rows = [];
  rows.push(checkRow({
    label: "Human authorship",
    tone: source.origin === "human_verified" ? "ok" : source.origin === "ai" ? "bad" : "todo",
    text: source.origin === "human_verified" ? "Verified by a human review."
      : source.origin === "ai" ? "The page discloses AI-generated text. It is excluded from trusted use."
        : "Not verified. Text stays out of the trusted collection until a person reviews authorship.",
    action: { label: source.origin === "human_verified" ? "Change" : "Review authorship", onclick: review("origin"), focusId: "check-origin" },
  }));
  rows.push(checkRow({
    label: "Copying permission",
    tone: source.permission === "text_permitted" ? "ok" : source.permission === "metadata_only" ? "info" : "todo",
    text: source.permission === "text_permitted" ? "Text collection is permitted."
      : source.permission === "metadata_only" ? "Metadata only. Text collection is off."
        : "Not reviewed. Free access is not a licence to copy.",
    action: { label: source.permission === "unknown" ? "Review permission" : "Change", onclick: review("permission"), focusId: "check-permission" },
  }));
  const access = ACCESS[source.access] || ACCESS.unknown;
  rows.push(checkRow({
    label: "Access",
    tone: source.access === "unknown" ? "todo" : source.access === "inaccessible" ? "bad" : source.access === "open" ? "ok" : "info",
    text: `${access.label}.${source.access_note ? ` ${source.access_note}` : ` ${access.meaning}`}`,
    action: { label: "Change access", onclick: review("access"), focusId: "check-access" },
  }));
  rows.push(checkRow({
    label: "Topic coverage",
    tone: source.topic_reviewed ? "ok" : "todo",
    text: source.topic_reviewed ? `Reviewed for ${topics}.` : `Candidate tags only (${topics}). Scores stay unknown until you confirm coverage against the material.`,
    action: { label: source.topic_reviewed ? "Change" : "Review topics", onclick: review("topics"), focusId: "check-topics" },
  }));
  const check = detail.update_check;
  rows.push(checkRow({
    label: "Editorial status",
    tone: summary.withdrawn || summary.pending_notices ? "bad" : "info",
    text: summary.withdrawn ? "A retraction is on record. This source is excluded from trusted text."
      : summary.pending_notices ? `${plural(detail.notices.filter((item) => item.notice.state === "reported").length || 1, "notice")} awaiting review. Trusted text is blocked until you decide.`
        : check.checked_at ? `No open notices. Last checked ${fmtRelative(check.checked_at)}; that does not prove the source is current.`
          : "No notices recorded and no editorial check yet. A quiet record is not proof the source is current.",
    action: { label: "Open notices", onclick: () => ctx.go("notices"), focusId: "check-notices" },
  }));
  const blockers = collectBlockers(source, summary);
  const collected = Boolean(source.collected_at) && source.word_count > 0;
  rows.push(checkRow({
    label: "Trusted text",
    tone: collected ? "ok" : summary.trusted_text_blocked ? "bad" : "todo",
    text: collected ? `${plural(source.word_count, "word")} collected on ${fmtDate(source.collected_at)}.`
      : summary.trusted_text_blocked ? "Blocked by an editorial notice or retraction."
        : blockers.length ? `No trusted text. Collecting needs ${blockers.join(", ")}.`
          : "Ready to collect permitted text.",
    action: !collected && blockers.length === 0 ? { label: "Collect text", onclick: (event) => runSourceAction("collect", source, event.currentTarget), focusId: "check-collect" } : null,
  }));
  return el("section", { class: "section", "aria-labelledby": "before-relying" },
    el("h3", { class: "tab-title", id: "before-relying" }, "Before relying on this source"),
    el("ul", { class: "checklist" }, rows));
}

export function renderOverviewTab(ctx) {
  const container = ctx.source.kind === "sources_of_sources";
  return el("div", { class: "tab-body" },
    checklist(ctx),
    el("section", { class: "section", "aria-labelledby": "scores-title" },
      el("h3", { class: "tab-title", id: "scores-title" }, "Scores by subject"),
      el("p", { class: "quiet" }, "Explainable, evidence-weighted estimates, not probabilities. A source’s reputation is not proof of any claim inside it."),
      container
        ? el("p", { class: "callout" }, icon("info"), el("span", {}, "A container’s reputation does not establish the accuracy of its contents, so Kumaru does not score it. Review the individual works it lists."))
        : ctx.detail.scores.map((score) => scoreCard({ score, formula: state.formula, source: ctx.source, level: 4 }))));
}

/* ─── Library at a glance ─── */

const TONES = { open: "ok", account: "accent", paywall: "warn", inaccessible: "err", unknown: "dim" };

function ledgerRow({ label, value, detail, onclick, focusId }) {
  const content = [el("span", { class: "ledger-label" }, label), el("span", { class: "ledger-value" }, fmtInt(value)), detail && el("span", { class: "ledger-detail" }, detail)];
  return el("li", {}, onclick
    ? el("button", { type: "button", class: "ledger-row", onclick, dataset: { focusId } }, content, icon("chevron", { size: 14 }))
    : el("div", { class: "ledger-row is-static" }, content));
}

function setFilter(patch) {
  Object.assign(state.filters, { access: "", kind: "", status: "", q: "", containerId: "", containerTitle: "", containerChildren: 0 }, patch);
  const search = document.getElementById("library-search");
  if (search) search.value = "";
  emit("filters-changed");
}

export function renderGlance() {
  const overview = state.overview;
  const queue = state.queue;
  const root = el("div", { class: "glance" });
  root.append(el("h2", { id: "inspector-title", class: "inspector-title", tabindex: "-1" }, "Library at a glance"));
  if (!overview) {
    root.append(el("p", { class: "quiet" }, "Counts appear once the catalogue has loaded."));
    return root;
  }
  const total = overview.total;
  root.append(el("p", { class: "lede" }, breadcrumb(state.topicId), el("span", {}, " · select a source to review it. Counts include everything under this subject.")));
  root.append(el("ul", { class: "ledger" },
    ledgerRow({ label: "Sources", value: total, onclick: () => setFilter({}), focusId: "glance-all" }),
    ledgerRow({ label: "Human verified", value: overview.by_origin.human_verified, onclick: () => setFilter({ status: "human_verified" }) }),
    ledgerRow({ label: "Needs review", value: overview.needs_review, onclick: () => setFilter({ status: "needs_review" }) }),
    ledgerRow({ label: "Pending notices", value: overview.pending_notices, onclick: () => navigate(viewHash("queue")) }),
    ledgerRow({ label: "Claims", value: overview.claims, detail: overview.claims ? `${fmtInt(overview.unreviewed_claims)} unreviewed` : "", onclick: overview.unreviewed_claims ? () => navigate(viewHash("queue")) : null })));
  if (total > 0) {
    const counts = ACCESS_ORDER.map((access) => [access, overview.by_access[access] || 0]).filter(([, count]) => count > 0);
    const summary = `Access of ${plural(total, "source")}: ${counts.map(([access, count]) => `${fmtInt(count)} ${ACCESS[access].label.toLowerCase()}`).join(", ")}.`;
    root.append(el("section", { class: "section", "aria-labelledby": "access-mix" },
      el("h3", { class: "tab-title", id: "access-mix" }, "How accessible is it?"),
      el("div", { class: "stack-bar", role: "img", "aria-label": summary },
        counts.map(([access, count]) => el("span", { class: `stack-seg tone-${TONES[access]}`, style: { flex: `${count} 1 0` } }))),
      el("ul", { class: "stack-legend" }, counts.map(([access, count]) => el("li", {},
        el("span", { class: `legend-dot tone-${TONES[access]}`, "aria-hidden": "true" }),
        el("span", {}, ACCESS[access].label), el("span", { class: "legend-count" }, fmtInt(count)))))));
  }
  const next = [button(`Discover sources for ${topicName(state.topicId)}`, { variant: "primary", iconName: "search", onclick: (event) => openDiscoverDialog(event.currentTarget) })];
  if (overview.needs_review > 0) next.push(button(`Review ${plural(overview.needs_review, "source")} awaiting review`, { onclick: () => setFilter({ status: "needs_review" }) }));
  if (queue && queue.pending_notices + queue.unreviewed_claims > 0) next.push(button("Open the review queue", { onclick: () => navigate(viewHash("queue")) }));
  root.append(el("section", { class: "section", "aria-labelledby": "next-actions" },
    el("h3", { class: "tab-title", id: "next-actions" }, "Next"),
    total === 0 ? el("p", { class: "quiet" }, `Nothing is catalogued under ${breadcrumbText(state.topicId)} yet. Discovery contacts an external provider and returns unreviewed candidates.`) : null,
    el("div", { class: "button-row" }, next)));
  return root;
}
