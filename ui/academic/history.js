// A human-readable timeline built from append-only review records, with the raw record
// one disclosure away for audits.
import { ACCESS, el, fmtDateTime, fmtUnit, ORIGIN, PERMISSION, plural } from "./dom.js";
import { topicName } from "./topics.js";

function sourceParts(review) {
  const parts = [];
  if (review.origin) parts.push(`Origin → ${ORIGIN[review.origin]?.label || review.origin}`);
  if (review.permission) parts.push(`Permission → ${PERMISSION[review.permission]?.label || review.permission}`);
  if (review.access) parts.push(`Access → ${ACCESS[review.access]?.label || review.access}`);
  if (review.access_note !== null && review.access_note !== undefined) parts.push(review.access_note ? "Access note recorded" : "Access note cleared");
  if (review.topic_ids) parts.push(`Topics → ${review.topic_ids.map(topicName).join(", ")}`);
  if (review.topic_reviewed === true) parts.push("Topic coverage reviewed");
  if (review.topic_reviewed === false) parts.push("Topic coverage revoked");
  if (review.evidence?.length) parts.push(plural(review.evidence.length, "finding"));
  return parts;
}

function noticeParts(review) {
  const parts = [`Notice → ${review.state}`];
  if (typeof review.impact === "number") parts.push(`impact ${fmtUnit(review.impact)}`);
  if (review.affected_topic_ids?.length) parts.push(review.affected_topic_ids.map(topicName).join(", "));
  return parts;
}

function claimParts(review) {
  const parts = [`Attribution → ${review.reviewed ? "Reviewed" : "Not reviewed"}`];
  if (review.evidence?.length) parts.push(plural(review.evidence.length, "finding"));
  return parts;
}

function describe(entry, kind) {
  const review = entry.review || {};
  if (kind === "claim") return { type: "Claim review", parts: claimParts(review), note: review.note };
  if (review.notice_id) return { type: "Notice review", parts: noticeParts(review), note: review.note };
  return { type: "Source review", parts: sourceParts(review), note: "" };
}

export function historyTimeline(entries, { kind = "source", empty = "No reviews recorded yet." } = {}) {
  if (!entries.length) return el("p", { class: "quiet" }, empty);
  return el("ol", { class: "timeline" }, [...entries].reverse().map((entry) => {
    const { type, parts, note } = describe(entry, kind);
    return el("li", { class: "timeline-item" },
      el("div", { class: "timeline-head" },
        el("time", { datetime: entry.recorded_at, class: "timeline-date" }, fmtDateTime(entry.recorded_at)),
        el("span", { class: "timeline-type" }, type)),
      el("p", { class: "timeline-summary" }, parts.length ? parts.join(" · ") : "No field changes recorded"),
      note && el("p", { class: "timeline-note" }, note),
      el("details", { class: "audit" }, el("summary", {}, "Audit record"),
        el("pre", { class: "audit-record" }, JSON.stringify(entry, null, 2))));
  }));
}
