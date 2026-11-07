// Evidence tab: findings grouped by dimension, plus the dialog that records or edits one.
import { api } from "./api.js";
import { openDialog, toast } from "./dialogs.js";
import {
  ACADEMIC_KINDS, badge, button, el, emptyState, EVIDENCE_HINTS, EVIDENCE_LABELS, EVIDENCE_ORDER, fmtDate, link, meter, plural,
} from "./dom.js";
import { areaField, checkField, rangeField, selectField, textField } from "./forms.js";
import { emit } from "./state.js";
import { topicName } from "./topics.js";

const NEEDS_TOPIC = new Set(["historical_acceptance", "proof", "factual_criticism"]);

function findingCard(item, ctx) {
  const editable = ACADEMIC_KINDS.includes(item.kind) && !item.id.startsWith("notice:");
  const label = EVIDENCE_LABELS[item.kind] || item.kind;
  return el("article", { class: "finding" },
    el("div", { class: "finding-head" },
      meter({ value: item.value, max: 1, text: item.value.toFixed(2), label: `${label} value ${item.value.toFixed(2)} out of 1` }),
      item.reviewed
        ? badge({ tone: "ok", iconName: "check", label: "Reviewed" })
        : badge({ tone: "dim", iconName: "question", label: "Unreviewed · not scored", variant: "dashed" }),
      el("span", { class: "tags" }, item.topic_ids.length
        ? item.topic_ids.map((id) => el("span", { class: "tag" }, topicName(id)))
        : el("span", { class: "tag" }, "All tagged subjects")),
      editable && button("Edit", {
        iconName: "edit", focusId: `edit-finding-${item.id}`,
        onclick: (event) => openFindingDialog({ source: ctx.source, finding: item, trigger: event.currentTarget }),
      }),
      !editable && item.id.startsWith("notice:") && el("span", { class: "finding-origin" }, "Set by a notice review")),
    el("p", { class: "finding-note" }, item.note),
    el("p", { class: "finding-meta" }, link(item.url), el("span", {}, ` · Recorded ${fmtDate(item.recorded_at)}`)));
}

export function renderEvidenceTab(ctx) {
  const evidence = ctx.source.evidence;
  const groups = EVIDENCE_ORDER.map((kind) => [kind, evidence.filter((item) => item.kind === kind)]).filter(([, items]) => items.length);
  return el("div", { class: "tab-body" },
    el("div", { class: "tab-head" },
      el("h3", { class: "tab-title" }, "Findings"),
      button("Add finding", { variant: "primary", iconName: "plus", focusId: "add-finding", onclick: (event) => openFindingDialog({ source: ctx.source, trigger: event.currentTarget }) })),
    el("p", { class: "quiet" }, "Only reviewed findings affect scores. A finding applies to the subject it names; evidence for one chapter never credits another."),
    groups.length === 0
      ? emptyState({ title: "No findings recorded yet", body: "Discovery is not verification. Record the evidence you have actually reviewed: proofs you checked, credentials, peer review, replication or flaws." })
      : groups.map(([kind, items]) => el("section", { class: "evidence-group", "aria-labelledby": `group-${kind}` },
        el("h4", { class: "group-title", id: `group-${kind}` }, EVIDENCE_LABELS[kind], el("span", { class: "group-count" }, plural(items.length, "finding"))),
        EVIDENCE_HINTS[kind] && el("p", { class: "group-hint" }, EVIDENCE_HINTS[kind]),
        items.map((item) => findingCard(item, ctx)))));
}

export async function openFindingDialog({ source, finding = null, trigger }) {
  const editing = Boolean(finding);
  const result = await openDialog({
    title: editing ? "Edit finding" : "Add finding", trigger, size: "lg", submitLabel: "Save finding",
    description: "Record what you actually reviewed. Unreviewed findings are kept but never affect a score.",
    build: () => {
      const kind = selectField({
        name: "kind", label: "Evidence dimension", value: finding?.kind || "proof", hint: EVIDENCE_HINTS[finding?.kind || "proof"],
        options: ACADEMIC_KINDS.map((value) => ({ value, label: EVIDENCE_LABELS[value] })),
      });
      const topic = selectField({
        name: "topic", label: "Subject", value: finding ? (finding.topic_ids[0] || "") : source.topic_ids[0],
        options: [{ value: "", label: "All tagged subjects (source-wide)" }, ...source.topic_ids.map((id) => ({ value: id, label: topicName(id) }))],
        hint: "Proof, historical acceptance and factual criticism must name one subject.",
      });
      const syncTopic = () => {
        const required = NEEDS_TOPIC.has(kind.control.value);
        const wide = topic.control.querySelector('option[value=""]');
        wide.disabled = required;
        if (required && topic.control.value === "") topic.control.value = source.topic_ids[0];
        kind.node.querySelector(".field-hint").textContent = EVIDENCE_HINTS[kind.control.value] || "";
      };
      kind.control.addEventListener("change", syncTopic);
      syncTopic();
      return [
        kind.node, topic.node,
        rangeField({ name: "value", label: "Assessed value (0–1)", value: finding?.value ?? 1, hint: "Your judgment of strength, or flaw severity for criticism. It is not a provider fact." }).node,
        textField({ name: "url", label: "Evidence URL", type: "url", required: true, value: finding?.url || "", placeholder: "https://" }).node,
        areaField({ name: "note", label: "What you checked, and its limits", required: true, rows: 4, maxlength: 4000, value: finding?.note || "" }).node,
        checkField({ name: "reviewed", label: "I personally reviewed this finding, not merely its link", checked: finding?.reviewed ?? false }).node,
      ];
    },
    onSubmit: (data) => api(`/sources/${source.id}/review`, {
      body: { evidence: [{
        ...(finding ? { id: finding.id } : {}),
        kind: String(data.get("kind")), value: Number(data.get("value")), url: String(data.get("url")).trim(),
        note: String(data.get("note")).trim(), topic_ids: data.get("topic") ? [String(data.get("topic"))] : [], reviewed: data.has("reviewed"),
      }] },
    }),
  });
  if (!result || result === true) return;
  emit("data-changed", { sourceId: source.id });
  toast({ message: editing ? "Finding updated." : "Finding recorded." });
}
