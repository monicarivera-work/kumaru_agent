// Claims tab: located assertions with their own status, score anatomy and findings, plus the
// dialogs to record a claim, review its attribution and attach a finding. Attribution review is
// kept separate from findings, so attaching proof never marks a claim as reviewed.
import { api } from "./api.js";
import { scoreCard } from "./anatomy.js";
import { openDialog, toast } from "./dialogs.js";
import {
  badge, button, CHALLENGE_KINDS, claimStatusBadge, el, emptyState, EVIDENCE_LABELS, fmtDate, fmtInt, link, meter, plural,
  SUPPORT_KINDS,
} from "./dom.js";
import { areaField, checkField, radioGroup, rangeField, selectField, textField } from "./forms.js";
import { historyTimeline } from "./history.js";
import { icon } from "./icons.js";
import { sourcePicker } from "./picker.js";
import { emit, gate, navigate, replaceHash, sourceHash, state } from "./state.js";
import { topicName } from "./topics.js";

const PAGE = 30;
const requests = gate();
let paging = { sourceId: "", offset: 0 };

const STANCES = {
  supports: { label: "Supports", icon: "up", tone: "ok" },
  challenges: { label: "Challenges", icon: "down", tone: "warn" },
  context: { label: "Context only", icon: "dot", tone: "dim" },
};

function findingRow(finding, detail, ctx, claim) {
  const stance = STANCES[finding.stance];
  const reference = detail.references[finding.source_id];
  return el("article", { class: "finding is-claim" },
    el("div", { class: "finding-head" },
      el("span", { class: `stance tone-${stance.tone}` }, icon(stance.icon), el("span", {}, stance.label)),
      el("span", { class: "finding-kind" }, EVIDENCE_LABELS[finding.kind] || finding.kind),
      meter({ value: finding.value, max: 1, text: finding.value.toFixed(2), label: `${EVIDENCE_LABELS[finding.kind]} value ${finding.value.toFixed(2)} out of 1` }),
      finding.reviewed ? badge({ tone: "ok", iconName: "check", label: "Reviewed" }) : badge({ tone: "dim", iconName: "question", label: "Unreviewed · not scored", variant: "dashed" }),
      button("Edit", { iconName: "edit", focusId: `edit-claim-finding-${finding.id}`, onclick: (event) => openClaimFindingDialog({ claim, ctx, detail, finding, trigger: event.currentTarget }) })),
    el("p", { class: "finding-ref" }, "From ",
      reference ? el("button", { type: "button", class: "text-link", onclick: () => navigate(sourceHash(reference.id)) }, reference.title) : "a source no longer in the catalogue",
      ` · ${finding.locator}`),
    el("p", { class: "finding-note" }, finding.note),
    el("p", { class: "finding-meta" }, link(finding.url)));
}

function claimBody(detail, ctx) {
  const { claim, assessment } = detail;
  return el("div", { class: "claim-body", id: `claim-body-${claim.id}` },
    el("dl", { class: "facts" },
      el("div", {}, el("dt", {}, "Location"), el("dd", {}, claim.locator)),
      el("div", {}, el("dt", {}, "Subject"), el("dd", {}, topicName(claim.topic_id))),
      el("div", {}, el("dt", {}, "Attribution"), el("dd", {}, claim.reviewed ? "Reviewed by you" : "Not reviewed")),
      el("div", {}, el("dt", {}, "Recorded"), el("dd", {}, fmtDate(claim.created_at)))),
    el("h4", { class: "group-title" }, "Assumptions"),
    claim.assumptions.length ? el("ul", { class: "assumptions" }, claim.assumptions.map((item) => el("li", {}, item))) : el("p", { class: "quiet" }, "No assumptions recorded."),
    el("p", { class: "status-line" }, claimStatusBadge(assessment.status), el("span", {}, "An assessment of the evidence recorded here, not a guarantee of truth.")),
    scoreCard({ score: assessment.score, formula: state.formula, source: ctx.source, level: 4 }),
    el("div", { class: "tab-head" },
      el("h4", { class: "group-title" }, "Findings", el("span", { class: "group-count" }, plural(claim.evidence.length, "finding"))),
      el("div", { class: "button-row" },
        button("Review attribution", { iconName: "check", focusId: `attribution-${claim.id}`, onclick: (event) => openAttributionDialog({ claim, trigger: event.currentTarget }) }),
        button("Attach finding", { variant: "primary", iconName: "plus", focusId: `attach-${claim.id}`, onclick: (event) => openClaimFindingDialog({ claim, ctx, detail, trigger: event.currentTarget }) }))),
    claim.evidence.length
      ? claim.evidence.map((finding) => findingRow(finding, detail, ctx, claim))
      : el("p", { class: "quiet" }, "No findings yet. Whole-source proof is not inherited by a claim; attach the evidence that applies to this exact assertion."),
    el("h4", { class: "group-title" }, "History"),
    historyTimeline(detail.reviews, { kind: "claim", empty: "No review recorded for this claim yet." }));
}

function claimCard(detail, ctx, open, toggle) {
  const { claim, assessment } = detail;
  const supports = claim.evidence.filter((item) => item.stance === "supports").length;
  const challenges = claim.evidence.filter((item) => item.stance === "challenges").length;
  return el("article", { class: ["claim", open && "is-open"] },
    el("button", {
      type: "button", class: "claim-head", "aria-expanded": open, "aria-controls": `claim-body-${claim.id}`, onclick: toggle,
      dataset: { claimHead: claim.id },
    },
    el("span", { class: ["claim-statement", open && "is-open"] }, claim.statement),
    el("span", { class: "claim-meta" },
      el("span", { class: "claim-locator" }, claim.locator),
      el("span", { class: "tag" }, topicName(claim.topic_id)),
      claimStatusBadge(assessment.status),
      el("span", { class: "stance-count tone-ok" }, icon("up"), el("span", {}, String(supports)), el("span", { class: "sr-only" }, "supporting findings")),
      el("span", { class: "stance-count tone-warn" }, icon("down"), el("span", {}, String(challenges)), el("span", { class: "sr-only" }, "challenging findings")))),
    open && claimBody(detail, ctx));
}

export function renderClaimsTab(ctx) {
  const host = el("div", { class: "tab-body" });
  if (ctx.container) {
    host.append(el("div", { class: "tab-head" }, el("h3", { class: "tab-title" }, "Claims")),
      emptyState({ title: "Claims belong to individual works", body: "A journal, library or repository is a catalogue. Record claims on the specific book, article or page that makes the assertion." }));
    return host;
  }
  if (paging.sourceId !== ctx.id) paging = { sourceId: ctx.id, offset: 0 };
  const list = el("div", { class: "claim-list", "aria-live": "polite" });
  const pager = el("div", { class: "pager" });
  let items = [];
  let selectedId = ctx.claimId || "";
  const total = ctx.summary.claim_count;

  const render = () => {
    list.replaceChildren(...items.map((detail) => claimCard(detail, ctx, detail.claim.id === selectedId, () => {
      const id = detail.claim.id;
      selectedId = selectedId === id ? "" : id;
      state.route.claimId = selectedId;
      replaceHash(sourceHash(ctx.id, "claims", selectedId));
      render();
      list.querySelector(`[data-claim-head="${CSS.escape(id)}"]`)?.focus();
    })));
    const first = paging.offset + 1;
    pager.replaceChildren(...(total > PAGE ? [
      el("span", { class: "pager-range" }, `${fmtInt(first)}–${fmtInt(Math.min(paging.offset + PAGE, total))} of ${fmtInt(total)}`),
      el("div", { class: "pager-buttons" },
        button("Previous", { disabled: paging.offset === 0, onclick: () => { paging.offset = Math.max(0, paging.offset - PAGE); load(); } }),
        button("Next", { disabled: paging.offset + PAGE >= total, onclick: () => { paging.offset += PAGE; load(); } })),
    ] : []));
  };

  async function load() {
    const token = requests.next();
    list.replaceChildren(el("p", { class: "quiet" }, "Loading claims…"));
    try {
      const page = total ? await api("/claims", { params: { source_id: ctx.id, limit: PAGE, offset: paging.offset } }) : [];
      let extra = null;
      if (selectedId && !page.some((detail) => detail.claim.id === selectedId)) {
        try { extra = await api(`/claims/${selectedId}`); } catch { extra = null; }
        if (extra && extra.claim.source_id !== ctx.id) extra = null;
      }
      if (!requests.current(token)) return;
      items = extra ? [extra, ...page] : page;
      if (selectedId && !items.some((detail) => detail.claim.id === selectedId)) selectedId = "";
      render();
      if (!items.length) {
        list.replaceChildren(emptyState({ title: "No claims recorded", body: "Capture an assertion, its exact location and its assumptions before attaching proof or criticism. A claim does not inherit the source’s reputation." }));
      }
    } catch (error) {
      if (!requests.current(token)) return;
      list.replaceChildren(emptyState({ tone: "error", title: "Claims could not be loaded", body: error.message, actions: [button("Try again", { onclick: load })] }));
    }
  }

  host.append(
    el("div", { class: "tab-head" },
      el("h3", { class: "tab-title" }, "Claims", el("span", { class: "tab-count" }, fmtInt(total))),
      button("Record claim", { variant: "primary", iconName: "plus", focusId: "record-claim", onclick: (event) => openRecordClaimDialog({ source: ctx.source, trigger: event.currentTarget }) })),
    el("p", { class: "quiet" }, "A source’s score does not prove every assertion inside it. Each claim keeps its own findings and status."),
    list, pager);
  load();
  return host;
}

/* ─── Dialogs ─── */

export async function openRecordClaimDialog({ source, trigger }) {
  const result = await openDialog({
    title: "Record claim", trigger, size: "lg", submitLabel: "Record claim",
    description: "The statement, location and assumptions cannot be edited later: a revised assertion is a new claim.",
    build: () => [
      areaField({ name: "statement", label: "Assertion or theorem", required: true, rows: 4, maxlength: 8000, hint: "As the source states it, in your own words if you must, never invented." }).node,
      selectField({ name: "topic_id", label: "Subject", value: source.topic_ids[0], options: source.topic_ids.map((id) => ({ value: id, label: topicName(id) })), required: true }).node,
      textField({ name: "locator", label: "Exact location", required: true, maxlength: 500, hint: "Page, section, theorem number or a stable anchor." }).node,
      areaField({ name: "assumptions", label: "Assumptions", rows: 3, hint: "One per line: the conditions under which the assertion holds." }).node,
    ],
    onSubmit: (data) => api("/claims", {
      body: {
        source_id: source.id, topic_id: String(data.get("topic_id")), statement: String(data.get("statement")).trim(),
        locator: String(data.get("locator")).trim(),
        assumptions: String(data.get("assumptions")).split("\n").map((line) => line.trim()).filter(Boolean),
      },
    }),
  });
  if (!result || result === true) return;
  emit("data-changed", { sourceId: source.id });
  navigate(sourceHash(source.id, "claims", result.claim.id), { replace: true });
  toast({ message: "Claim recorded as unreviewed. Review its attribution, then attach the evidence that applies to it." });
}

export async function openAttributionDialog({ claim, trigger }) {
  const result = await openDialog({
    title: "Review claim attribution", trigger, size: "md", submitLabel: "Save review",
    description: "Confirm that the claim is correctly attributed. This is separate from the findings attached to it.",
    build: () => [
      checkField({ name: "reviewed", label: "I checked this claim’s attribution, subject, location and assumptions", checked: claim.reviewed }).node,
      areaField({ name: "note", label: "Review note", required: true, rows: 3, maxlength: 4000, hint: "Kept in the claim’s audit history." }).node,
    ],
    onSubmit: (data) => api(`/claims/${claim.id}/review`, { body: { reviewed: data.has("reviewed"), note: String(data.get("note")).trim() } }),
  });
  if (!result || result === true) return;
  emit("data-changed", { sourceId: claim.source_id });
  toast({ message: "Claim review saved." });
}

export async function openClaimFindingDialog({ claim, ctx, detail, finding = null, trigger }) {
  const editing = Boolean(finding);
  const own = { id: ctx.source.id, title: ctx.source.title, kind: ctx.source.kind, access: ctx.source.access, origin: ctx.source.origin };
  const result = await openDialog({
    title: editing ? "Edit claim finding" : "Attach finding", trigger, size: "lg", submitLabel: "Save finding",
    description: "A finding applies to this exact claim. It does not change whether the claim’s attribution is reviewed.",
    build: () => {
      const stance = radioGroup({
        name: "stance", legend: "Relationship to the claim", value: finding?.stance || "supports", layout: "inline",
        options: [
          { value: "supports", label: "Supports", hint: "Evidence for the claim." },
          { value: "challenges", label: "Challenges", hint: "A specific factual flaw, or a retraction." },
          { value: "context", label: "Context only", hint: "Shown, but it does not score." },
        ],
      });
      const kind = selectField({ name: "kind", label: "Evidence dimension", value: finding?.kind || "proof", options: [] });
      let desired = finding?.kind || "proof";
      kind.control.addEventListener("change", () => { desired = kind.control.value; });
      const syncKinds = () => {
        const allowed = stance.value() === "challenges" ? CHALLENGE_KINDS : SUPPORT_KINDS;
        if (!allowed.includes(desired)) desired = allowed[0];
        kind.control.replaceChildren(...allowed.map((value) => el("option", { value, selected: value === desired }, EVIDENCE_LABELS[value])));
      };
      stance.inputs.forEach((input) => input.addEventListener("change", syncKinds));
      syncKinds();
      const source = sourcePicker({
        name: "source_id", label: "Referenced source", required: true, initial: finding ? (detail.references[finding.source_id] || own) : own,
        hint: "Its authorship and topic coverage must be reviewed before it counts.",
      });
      return [
        stance.node, kind.node, source.node,
        rangeField({ name: "value", label: "Assessed strength or flaw severity (0–1)", value: finding?.value ?? 1 }).node,
        textField({ name: "url", label: "Evidence URL", type: "url", required: true, value: finding?.url || "", placeholder: "https://" }).node,
        textField({ name: "locator", label: "Exact location in that source", required: true, maxlength: 500, value: finding?.locator || "" }).node,
        areaField({ name: "note", label: "Proof, assumptions or the specific factual flaw", required: true, rows: 4, maxlength: 4000, value: finding?.note || "" }).node,
        checkField({ name: "reviewed", label: "I reviewed this evidence and how it applies to this exact claim", checked: finding?.reviewed ?? false }).node,
      ];
    },
    onSubmit: (data) => api(`/claims/${claim.id}/review`, {
      body: {
        reviewed: claim.reviewed, note: String(data.get("note")).trim(),
        evidence: [{
          ...(finding ? { id: finding.id } : {}),
          source_id: String(data.get("source_id")), stance: String(data.get("stance")), kind: String(data.get("kind")),
          value: Number(data.get("value")), url: String(data.get("url")).trim(), locator: String(data.get("locator")).trim(),
          note: String(data.get("note")).trim(), topic_ids: [claim.topic_id], reviewed: data.has("reviewed"),
        }],
      },
    }),
  });
  if (!result || result === true) return;
  emit("data-changed", { sourceId: claim.source_id });
  toast({ message: editing ? "Finding updated." : "Finding attached. Attribution review is unchanged." });
}
