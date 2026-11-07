// Notices tab: editorial reports for this source (corrections, concerns, retractions), the
// last-check line, and the dialogs to report a notice or record a human decision on it.
import { api } from "./api.js";
import { openDialog, toast } from "./dialogs.js";
import {
  badge, button, el, emptyState, fmtDateTime, fmtEventDate, fmtInt, fmtRelative, fmtUnit, link, NOTICE_KINDS, NOTICE_STATES, plural,
} from "./dom.js";
import { areaField, radioGroup, rangeField, selectField, textField } from "./forms.js";
import { emit, gate, navigate, sourceHash } from "./state.js";
import { topicName } from "./topics.js";

const PAGE = 30;
const requests = gate();
let paging = { sourceId: "", offset: 0 };

function checkLine(check, source) {
  if (!check.checked_at) {
    return { tone: "", text: source.doi && source.kind !== "sources_of_sources"
      ? "No editorial check recorded yet. Use “Check editorial updates” to ask Crossref."
      : "No editorial check recorded. Crossref monitoring needs a DOI, so report notices you find yourself." };
  }
  const when = `${fmtRelative(check.checked_at)} (${fmtDateTime(check.checked_at)})`;
  if (check.error) return { tone: "is-err", text: `The last check failed ${when}: ${check.error}` };
  if (check.partial) return { tone: "is-warn", text: `Last checked ${when}. Partial scan: only the first 100 update records were examined.` };
  return { tone: "", text: `Last checked ${when}. A check that reports nothing does not prove the source is current.` };
}

function snippet(text) {
  return text.length > 110 ? `${text.slice(0, 107)}…` : text;
}

function noticeCard(item, ctx) {
  const { notice } = item;
  const info = NOTICE_STATES[notice.state];
  return el("article", { class: ["notice", `is-${notice.state}`] },
    el("div", { class: "notice-head" },
      el("span", { class: "tag" }, NOTICE_KINDS[notice.kind] || notice.kind),
      badge({ tone: info.tone, iconName: info.icon, label: info.label, variant: info.variant }),
      el("span", { class: "notice-provider" }, notice.provider === "crossref" ? "Crossref" : "Reported manually")),
    el("h4", { class: "notice-title" }, notice.title),
    el("p", { class: "notice-meta" },
      notice.event_date && el("span", {}, `${fmtEventDate(notice.event_date)} · `), link(notice.url)),
    notice.state === "confirmed" && el("p", { class: "notice-impact" },
      `Impact ${fmtUnit(notice.impact)}`, notice.affected_topic_ids.length ? ` on ${notice.affected_topic_ids.map(topicName).join(", ")}` : " on the whole source"),
    notice.review_note && el("p", { class: "notice-note" }, notice.review_note),
    item.affected_claims.length > 0 && el("section", { class: "notice-links" },
      el("h5", { class: "mini-title" }, "Affected claims"),
      el("ul", {}, item.affected_claims.map((claim) => el("li", {},
        el("button", { type: "button", class: "text-link", onclick: () => navigate(sourceHash(claim.source_id, "claims", claim.id)) }, snippet(claim.statement))))),
      ),
    item.dependent_source_ids.length > 0 && el("section", { class: "notice-links" },
      el("h5", { class: "mini-title" }, "Sources that cite or are corroborated by this one"),
      el("ul", {}, item.dependent_source_ids.map((id) => el("li", {},
        el("button", { type: "button", class: "text-link", onclick: () => navigate(sourceHash(id)) }, item.dependent_sources[id]?.title || "Source no longer in the catalogue"))))),
    el("div", { class: "button-row" },
      button("Review notice", { variant: notice.state === "reported" ? "primary" : "secondary", iconName: "check", focusId: `review-notice-${notice.id}`,
        onclick: (event) => openNoticeReviewDialog({ item, source: ctx.source, trigger: event.currentTarget }) })));
}

export function renderNoticesTab(ctx) {
  if (paging.sourceId !== ctx.id) paging = { sourceId: ctx.id, offset: 0 };
  const total = ctx.summary.notice_count;
  const check = checkLine(ctx.detail.update_check, ctx.source);
  const list = el("div", { class: "notice-list", "aria-live": "polite" });
  const pager = el("div", { class: "pager" });

  const renderPager = () => pager.replaceChildren(...(total > PAGE ? [
    el("span", { class: "pager-range" }, `${fmtInt(paging.offset + 1)}–${fmtInt(Math.min(paging.offset + PAGE, total))} of ${fmtInt(total)}`),
    el("div", { class: "pager-buttons" },
      button("Previous", { disabled: paging.offset === 0, onclick: () => { paging.offset = Math.max(0, paging.offset - PAGE); load(); } }),
      button("Next", { disabled: paging.offset + PAGE >= total, onclick: () => { paging.offset += PAGE; load(); } })),
  ] : []));

  async function load() {
    const token = requests.next();
    try {
      const items = paging.offset === 0
        ? ctx.detail.notices.slice(0, PAGE)
        : await api("/notices", { params: { source_id: ctx.id, limit: PAGE, offset: paging.offset } });
      if (!requests.current(token)) return;
      list.replaceChildren(...(items.length ? items.map((item) => noticeCard(item, ctx))
        : [emptyState({ title: "No editorial notices recorded", body: "Notices reported by Crossref or by you appear here. An empty list is not proof that no correction exists." })]));
      renderPager();
    } catch (error) {
      if (!requests.current(token)) return;
      list.replaceChildren(emptyState({ tone: "error", title: "Notices could not be loaded", body: error.message, actions: [button("Try again", { onclick: load })] }));
    }
  }

  const host = el("div", { class: "tab-body" },
    el("div", { class: "tab-head" },
      el("h3", { class: "tab-title" }, "Editorial notices", el("span", { class: "tab-count" }, fmtInt(total))),
      button("Report notice", { variant: "primary", iconName: "plus", focusId: "report-notice", onclick: (event) => openReportNoticeDialog({ source: ctx.source, trigger: event.currentTarget }) })),
    el("p", { class: ["check-line", check.tone] }, check.text),
    list, pager);
  load();
  return host;
}

/* ─── Dialogs ─── */

export async function openReportNoticeDialog({ source, trigger }) {
  const result = await openDialog({
    title: "Report editorial notice", trigger, size: "md", submitLabel: "Report notice",
    description: "Use this for a correction, concern or retraction you found yourself. A reported notice blocks trusted text for this source until you review it.",
    build: () => [
      textField({ name: "title", label: "Notice title", required: true, maxlength: 2000 }).node,
      selectField({ name: "kind", label: "Notice type", value: "correction", options: Object.entries(NOTICE_KINDS).map(([value, label]) => ({ value, label })) }).node,
      textField({ name: "url", label: "Official notice URL", type: "url", required: true, placeholder: "https://" }).node,
      textField({ name: "event_date", label: "Event date (optional)", type: "date", hint: "Leave empty if the notice does not give one." }).node,
    ],
    onSubmit: (data) => api("/notices", {
      body: {
        source_id: source.id, title: String(data.get("title")).trim(), kind: String(data.get("kind")),
        url: String(data.get("url")).trim(), event_date: String(data.get("event_date")) || null,
      },
    }),
  });
  if (!result || result === true) return;
  emit("data-changed", { sourceId: source.id });
  toast({ message: "Notice reported. It stays pending, and trusted text is blocked, until you review it." });
}

export async function openNoticeReviewDialog({ item, source, trigger }) {
  const { notice } = item;
  const retraction = notice.kind === "retraction";
  const result = await openDialog({
    title: "Review notice", trigger, size: "lg", submitLabel: "Save decision",
    description: notice.title,
    build: () => {
      const decision = radioGroup({
        name: "state", legend: "Decision", value: notice.state,
        options: [
          { value: "reported", label: "Reported", hint: "Still needs review. Keeps trusted text blocked." },
          { value: "confirmed", label: "Confirmed", hint: "The problem is real and affects this source." },
          { value: "dismissed", label: "Dismissed", hint: "Not a real problem for this source (for example, the wrong work)." },
        ],
      });
      const impact = rangeField({
        name: "impact", label: "Assessed impact (0–1)", value: notice.impact ?? (retraction ? 1 : 0), disabled: retraction,
        hint: retraction ? "A confirmed retraction always has impact 1." : "How much the flaw undermines the work. Zero acknowledges a non-factual correction.",
      });
      const topics = el("fieldset", { class: "field" },
        el("legend", { class: "field-label" }, "Affected subjects"),
        el("p", { class: "field-hint" }, retraction ? "A retraction applies to the whole source." : "Required when a factual impact is above zero."),
        el("div", { class: "choice-list" }, source.topic_ids.map((id) => el("label", { class: "checkbox" },
          el("input", { type: "checkbox", class: "check-input", name: "affected_topic_ids", value: id, checked: notice.affected_topic_ids.includes(id), disabled: retraction }),
          el("span", { class: "check-text" }, topicName(id))))));
      const conditional = el("div", { class: "conditional" }, impact.node, topics);
      const warning = el("p", { class: "callout is-warn" }, "Confirming clears any collected text and can change scores. Nothing is applied until you save.");
      const sync = () => {
        const confirmed = decision.value() === "confirmed";
        conditional.hidden = !confirmed;
        warning.hidden = !confirmed;
        if (!retraction) {
          impact.control.disabled = impact.slider.disabled = !confirmed;
          topics.querySelectorAll("input").forEach((input) => { input.disabled = !confirmed; });
        }
      };
      decision.inputs.forEach((input) => input.addEventListener("change", sync));
      sync();
      return [
        decision.node, conditional, warning,
        areaField({ name: "note", label: "What you verified, or why you dismissed it", required: true, rows: 3, maxlength: 4000, value: notice.review_note || "" }).node,
      ];
    },
    onSubmit: async (data) => {
      const confirmed = data.get("state") === "confirmed";
      const impact = retraction ? 1 : Number(data.get("impact"));
      const affected = confirmed && !retraction ? data.getAll("affected_topic_ids").map(String) : [];
      if (confirmed && !retraction && impact > 0 && !affected.length) {
        throw new Error("Choose the subjects this affects, or set the impact to 0 for a non-factual correction.");
      }
      return api(`/notices/${notice.id}/review`, {
        body: { state: String(data.get("state")), note: String(data.get("note")).trim(), impact: confirmed ? impact : null, affected_topic_ids: affected },
      });
    },
  });
  if (!result || result === true) return;
  emit("data-changed", { sourceId: source.id });
  toast({ message: result.notice.state === "confirmed" ? `Notice confirmed. ${plural(item.affected_claims.length, "claim")} may need another look.` : "Notice decision saved." });
}
