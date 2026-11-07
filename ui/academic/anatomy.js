// The score anatomy: a weighted bar showing which evidence dimensions earned credit, which are
// missing (hatched) and which penalties apply. The headline numbers are always the server's.
import { el, EVIDENCE_LABELS, isNumber, uid } from "./dom.js";
import { icon } from "./icons.js";
import { breadcrumb } from "./topics.js";

const USEFULNESS_LABELS = {
  legitimacy: "Legitimacy share", citations: "Citations", references: "References",
  independent_sources: "Independent corroboration", content_length: "Topic text",
};

const points = (value) => value.toFixed(1);

function segmentBar(segments, label) {
  return el("div", { class: "anatomy-bar", role: "img", "aria-label": label },
    segments.map((segment) => el("span", {
      class: ["seg", segment.missing && "is-missing", segment.empty && "is-empty"], style: { flex: `${segment.weight} 1 0` },
    }, segment.fill > 0 && el("span", { class: "seg-fill", style: { width: `${Math.min(100, segment.fill * 100).toFixed(1)}%` } }))));
}

function legendRow(segment) {
  return el("li", { class: ["legend-row", segment.missing && "is-missing"] },
    el("span", { class: ["legend-swatch", segment.missing && "is-missing", segment.empty && "is-empty"], "aria-hidden": "true" }),
    el("span", { class: "legend-name" }, segment.label),
    el("span", { class: "legend-points" }, `${points(segment.weight)} pts`),
    el("span", { class: "legend-value" }, segment.text));
}

function summary(title, segments, known) {
  const parts = segments.map((segment) => (segment.missing
    ? `${segment.label}: ${known ? segment.reason : "not scored"}`
    : `${segment.label}: ${points(segment.fill * segment.weight)} of ${points(segment.weight)} points`));
  return `${title}. ${parts.join("; ")}.`;
}

function legitimacySegments(score, formula) {
  const scored = isNumber(score.legitimacy);
  return Object.entries(formula.legitimacy_weights).map(([key, weight]) => {
    const value = score.components?.[key];
    const missing = !scored || !isNumber(value);
    const fill = missing ? 0 : value;
    return {
      key, label: EVIDENCE_LABELS[key] || key, weight: weight * 100, fill, missing,
      reason: "no reviewed evidence",
      text: missing ? (scored ? "no reviewed evidence" : "not scored")
        : `${value.toFixed(2)} × ${points(weight * 100)} = ${points(value * weight * 100)}`,
    };
  });
}

function usefulnessSegments(score, formula, source) {
  const scored = isNumber(score.usefulness);
  const reasons = {
    citations: source?.citation_count === null ? "not reported by the provider" : "",
    references: source?.reference_count === null ? "not reported by the provider" : "",
    independent_sources: "no independent corroboration recorded",
    content_length: "no collected topic text",
  };
  return Object.entries(formula.usefulness_points).map(([key, weight]) => {
    const value = key === "legitimacy" ? (scored ? score.legitimacy / 100 : null) : score.components?.[key];
    const unknown = !scored || !isNumber(value);
    const hatch = unknown || (key !== "legitimacy" && value === 0 && reasons[key] !== "");
    const reason = scored ? reasons[key] || "none" : "not scored";
    return {
      key, label: USEFULNESS_LABELS[key] || key, weight, fill: unknown ? 0 : value, missing: hatch,
      empty: !hatch && value === 0, reason,
      text: hatch ? reason : `${value.toFixed(2)} × ${points(weight)} = ${points(value * weight)}`,
    };
  });
}

function penalties(score) {
  const criticism = score.components?.factual_criticism;
  const retraction = score.components?.retraction;
  const notes = [];
  if (isNumber(criticism) && criticism > 0) {
    notes.push({ tone: "warn", text: `Factual criticism ${criticism.toFixed(2)}: credit × ${(1 - 0.8 * criticism).toFixed(2)}, from × (1 − 0.8·c).` });
  }
  if (isNumber(retraction) && retraction > 0) {
    notes.push({ tone: "err", text: `Retraction ${retraction.toFixed(2)}: legitimacy capped at ${(10 * (1 - retraction)).toFixed(1)}, from 10·(1 − r).` });
  }
  return notes;
}

function headlineNumber(label, value, note = "/ 100") {
  const known = isNumber(value);
  return el("div", { class: ["headline", !known && "is-unknown"] },
    el("span", { class: "headline-label" }, label),
    el("span", { class: "headline-value" }, known ? value.toFixed(1) : "Unknown", known && el("span", { class: "headline-unit" }, note)));
}

export function scoreCard({ score, formula, source, level = 4 }) {
  const headingId = uid("score");
  const scored = isNumber(score.legitimacy);
  const legit = legitimacySegments(score, formula);
  const useful = usefulnessSegments(score, formula, source);
  const notes = penalties(score);
  const reason = score.warnings?.[0] || "No reviewed academic evidence; scores are unknown.";
  const remaining = scored ? score.warnings || [] : (score.warnings || []).slice(1);
  return el("article", { class: ["score-card", !scored && "is-unscored"], "aria-labelledby": headingId },
    el(`h${level}`, { class: "score-topic", id: headingId }, breadcrumb(score.topic_id)),
    el("div", { class: "headlines" },
      headlineNumber("Legitimacy", score.legitimacy),
      headlineNumber("Usefulness", score.usefulness),
      el("div", { class: "headline" },
        el("span", { class: "headline-label" }, "Evidence coverage"),
        el("span", { class: "headline-value" }, `${Math.round((score.evidence_coverage || 0) * 100)}`, el("span", { class: "headline-unit" }, "%")))),
    !scored && el("p", { class: "unscored-reason" }, icon("info"), el("span", {}, reason)),
    el("section", { class: "anatomy", "aria-label": "Legitimacy anatomy" },
      el("h5", { class: "anatomy-title" }, "Legitimacy anatomy"),
      segmentBar(legit, summary("Legitimacy anatomy", legit, scored)),
      el("ul", { class: "legend" }, legit.map(legendRow)),
      scored && (notes.length
        ? el("ul", { class: "penalties" }, notes.map((note) => el("li", { class: `is-${note.tone}` }, icon("alert"), el("span", {}, note.text))))
        : el("p", { class: "anatomy-quiet" }, "No factual criticism or retraction recorded."))),
    el("section", { class: "anatomy", "aria-label": "Usefulness anatomy" },
      el("h5", { class: "anatomy-title" }, "Usefulness anatomy"),
      segmentBar(useful, summary("Usefulness anatomy", useful, scored)),
      el("ul", { class: "legend" }, useful.map(legendRow))),
    remaining.length > 0 && el("ul", { class: "score-warnings" }, remaining.map((text) => el("li", {}, text))),
    el("p", { class: "score-foot" }, `Formula ${score.version || formula.version} · explainable estimate, not a probability of truth`));
}
