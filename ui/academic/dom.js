// DOM construction, labels, formatting and the small visual components shared by every view.
// Nothing here parses markup: all dynamic strings become text nodes or attribute values.
import { icon } from "./icons.js";

const PROPERTIES = new Set(["value", "checked", "selected", "disabled", "required", "multiple", "hidden", "open", "readOnly"]);

export function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes || {})) {
    if (value === undefined || value === null) continue;
    if (name.startsWith("on")) {
      // Never turn a string into an inline handler attribute.
      if (typeof value === "function") node.addEventListener(name.slice(2), value);
      continue;
    }
    if (name === "class") {
      const names = [value].flat(Infinity).filter(Boolean).join(" ");
      if (names) node.className = names;
    } else if (name === "dataset") Object.assign(node.dataset, value);
    else if (name === "style") {
      for (const [key, item] of Object.entries(value)) node.style.setProperty(key, String(item));
    } else if (PROPERTIES.has(name)) node[name] = value;
    else if (value === false && !name.startsWith("aria-")) continue;
    else node.setAttribute(name, value === true && !name.startsWith("aria-") ? "" : String(value));
  }
  append(node, children);
  return node;
}

export function append(node, children) {
  for (const child of [children].flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node, ...children) {
  node.replaceChildren();
  return append(node, children);
}

let counter = 0;
export const uid = (prefix = "ac") => `${prefix}-${++counter}`;

// Only http(s) URLs become links; anything else is shown as inert text.
export function link(url, label = url, attributes = {}) {
  let parsed = null;
  try { parsed = new URL(url); } catch { /* shown as text below */ }
  if (!parsed || !["https:", "http:"].includes(parsed.protocol)) {
    return el("span", { class: "unlinked", title: "Not a web address" }, label);
  }
  return el("a", { class: "ext-link", ...attributes, href: parsed.href, target: "_blank", rel: "noopener noreferrer" },
    el("span", {}, label), icon("external", { size: 12 }), el("span", { class: "sr-only" }, " (opens in a new tab)"));
}

export const plural = (count, one, many = `${one}s`) => `${fmtInt(count)} ${count === 1 ? one : many}`;

export async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch { /* clipboard needs a secure context; try the legacy path */ }
  const area = el("textarea", { readOnly: true, class: "sr-only", "aria-hidden": "true" });
  area.value = value;
  document.body.append(area);
  area.select();
  let copied = false;
  try { copied = document.execCommand("copy"); } catch { /* reported by the caller */ }
  area.remove();
  return copied;
}

export function focusById(focusId) {
  const node = document.querySelector(`[data-focus-id="${CSS.escape(focusId)}"]`);
  if (node) node.focus({ preventScroll: false });
  return Boolean(node);
}

/* ─── Vocabulary ─── */

export const ACCESS = {
  open: { label: "Immediately accessible", short: "Accessible", icon: "unlock", tone: "ok",
    meaning: "The landing page was reachable without signing in. This is not a reuse licence." },
  account: { label: "Account required", short: "Account required", icon: "user", tone: "accent",
    meaning: "A login sits between the reader and the content." },
  paywall: { label: "Paywall", short: "Paywall", icon: "lock", tone: "warn",
    meaning: "Reading requires payment or a subscription." },
  inaccessible: { label: "Application inaccessible", short: "Inaccessible", icon: "ban", tone: "err",
    meaning: "Kumaru could not retrieve it: robots refusal, a browser challenge, an unsupported format or a dead link." },
  unknown: { label: "Unknown", short: "Access unknown", icon: "question", tone: "dim",
    meaning: "Access has not been established yet." },
};
export const ACCESS_ORDER = ["open", "account", "paywall", "inaccessible", "unknown"];

export const ORIGIN = {
  human_verified: { label: "Human verified", icon: "check", tone: "ok", variant: "solid" },
  unknown: { label: "Not verified", icon: "question", tone: "dim", variant: "dashed" },
  ai: { label: "AI disclosed", icon: "alert", tone: "err", variant: "outline" },
};

export const PERMISSION = {
  text_permitted: { label: "Text permitted", icon: "check", tone: "ok", variant: "solid" },
  metadata_only: { label: "Metadata only", icon: "info", tone: "dim", variant: "solid" },
  unknown: { label: "Permission not reviewed", icon: "question", tone: "dim", variant: "dashed" },
};

export const KINDS = {
  article: { label: "Article", icon: "article" },
  book: { label: "Book", icon: "book" },
  study: { label: "Study", icon: "study" },
  website: { label: "Website", icon: "globe" },
  sources_of_sources: { label: "Source of Sources", icon: "library" },
};

export const CLAIM_STATUS = {
  evidence_supported: { label: "Evidence supported", icon: "check", tone: "ok", variant: "solid" },
  disputed: { label: "Disputed", icon: "alert", tone: "warn", variant: "solid" },
  withdrawn: { label: "Withdrawn", icon: "ban", tone: "err", variant: "solid" },
  insufficient_evidence: { label: "Insufficient evidence", icon: "info", tone: "dim", variant: "solid" },
  unreviewed: { label: "Unreviewed", icon: "question", tone: "dim", variant: "dashed" },
};

export const NOTICE_KINDS = {
  retraction: "Retraction", correction: "Correction", expression_of_concern: "Expression of concern", update: "Editorial update",
};
export const NOTICE_STATES = {
  reported: { label: "Reported · needs review", icon: "alert", tone: "warn", variant: "solid" },
  confirmed: { label: "Confirmed", icon: "check", tone: "err", variant: "outline" },
  dismissed: { label: "Dismissed", icon: "x", tone: "dim", variant: "solid" },
};

export const EVIDENCE_ORDER = [
  "historical_acceptance", "credentials", "proof", "peer_review", "replication", "corroboration",
  "factual_criticism", "retraction", "origin", "permission",
];
export const EVIDENCE_LABELS = {
  historical_acceptance: "Historical acceptance", credentials: "Credentials", proof: "Proof",
  peer_review: "Peer review", replication: "Replication", corroboration: "Corroboration",
  factual_criticism: "Factual criticism", retraction: "Retraction", origin: "Origin", permission: "Permission",
};
export const EVIDENCE_HINTS = {
  historical_acceptance: "How long and how widely this result has been accepted within the topic.",
  credentials: "Relevant qualifications of the author for this topic.",
  proof: "A proof or derivation you checked, with its assumptions.",
  peer_review: "Evidence of independent review before publication.",
  replication: "Independent confirmation of the result.",
  corroboration: "A recorded agreement. Scores use explicit “corroborates” links between sources.",
  factual_criticism: "A specific factual flaw you reviewed. Its severity scales the score down.",
  retraction: "A confirmed withdrawal. It caps the score.",
};
export const ACADEMIC_KINDS = EVIDENCE_ORDER.slice(0, 8);
export const SUPPORT_KINDS = ["proof", "historical_acceptance", "replication", "peer_review", "credentials", "corroboration"];
export const CHALLENGE_KINDS = ["factual_criticism", "retraction"];

export const LEVELS = [
  { id: "area", label: "Area" }, { id: "subfield", label: "Subfield" }, { id: "unit", label: "Unit" },
  { id: "chapter", label: "Chapter" }, { id: "subchapter", label: "Sub-chapter" },
];
export const levelLabel = (id) => LEVELS.find((level) => level.id === id)?.label || id;

export const RELATION_LABELS = {
  contains: "Contains", cites: "Cites", corroborates: "Corroborates", criticizes: "Criticizes", related: "Related",
  open_copy: "Free copy",
};

// What an open-access index reports. Metadata for a reviewer: a license is never an approval.
export const OA_STATUS = {
  gold: { label: "Gold — open at the publisher", short: "Gold OA" },
  diamond: { label: "Diamond — open, no author fees", short: "Diamond OA" },
  hybrid: { label: "Hybrid — open article in a paid journal", short: "Hybrid OA" },
  bronze: { label: "Bronze — free to read, no license", short: "Bronze OA" },
  green: { label: "Green — free copy in a repository", short: "Green OA" },
  closed: { label: "Closed — no free copy known", short: "No free copy known" },
};
export const OA_VERSIONS = {
  publishedVersion: "Published version",
  acceptedVersion: "Accepted manuscript (peer reviewed, not typeset)",
  submittedVersion: "Submitted preprint (not yet peer reviewed)",
};
export const OA_PROVIDERS = { openalex: "OpenAlex", unpaywall: "Unpaywall" };

export function licenseLabel(value) {
  const license = String(value || "").trim().toLowerCase();
  if (!license) return "No license stated — free to read, not to reuse";
  if (["cc0", "public-domain", "pd"].includes(license)) return "Public domain — reuse allowed";
  if (license === "cc-by") return "CC BY — reuse with attribution";
  if (license === "cc-by-sa") return "CC BY-SA — reuse with attribution, share alike";
  if (license.startsWith("cc-by-nc") || license.startsWith("cc-by-nd")) {
    return `${license.toUpperCase()} — reuse restricted (non-commercial or no derivatives)`;
  }
  return `Other license (${String(value).trim()}) — check its terms`;
}

export const pretty = (value) => String(value || "").replaceAll("_", " ");

/* ─── Formatting ─── */

const intFormat = new Intl.NumberFormat();
const dateFormat = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" });
const dateTimeFormat = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const relativeFormat = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

export const fmtInt = (value) => intFormat.format(value ?? 0);

function parse(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}
export const fmtDate = (value) => { const date = parse(value); return date ? dateFormat.format(date) : ""; };
export const fmtDateTime = (value) => { const date = parse(value); return date ? dateTimeFormat.format(date) : ""; };
export function fmtRelative(value) {
  const date = parse(value);
  if (!date) return "";
  const seconds = (date.getTime() - Date.now()) / 1000;
  for (const [unit, size] of [["year", 31536000], ["month", 2592000], ["day", 86400], ["hour", 3600], ["minute", 60]]) {
    if (Math.abs(seconds) >= size) return relativeFormat.format(Math.round(seconds / size), unit);
  }
  return "just now";
}
// Crossref event dates can be a bare year or year-month; do not invent a day.
export const fmtEventDate = (value) => (/^\d{4}-\d{2}-\d{2}$/.test(value || "") ? fmtDate(value) : value || "");

export const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
export const fmtScore = (value) => (isNumber(value) ? value.toFixed(1) : "Unknown");
export const fmtUnit = (value) => (isNumber(value) ? value.toFixed(2) : "—");

export function authorsLine(authors = [], count = authors.length) {
  if (!count) return "";
  if (count === 1) return authors[0];
  if (count === 2 && authors.length >= 2) return `${authors[0]} & ${authors[1]}`;
  return `${authors[0]} et al.`;
}

/* ─── Components ─── */

export function button(label, { variant = "secondary", iconName, onclick, type = "button", disabled, title, focusId, describedBy, ...rest } = {}) {
  return el("button", {
    type, class: ["btn", `btn-${variant}`], onclick, disabled, title, "aria-describedby": describedBy,
    dataset: focusId ? { focusId } : undefined, ...rest,
  }, iconName && icon(iconName), label && el("span", {}, label));
}

export function iconButton(iconName, label, { onclick, focusId, ...rest } = {}) {
  return el("button", {
    type: "button", class: "btn btn-icon", "aria-label": label, title: label, onclick,
    dataset: focusId ? { focusId } : undefined, ...rest,
  }, icon(iconName));
}

export function badge({ tone = "dim", iconName, label, title, variant = "solid", srPrefix = "" }) {
  return el("span", { class: ["badge", `tone-${tone}`, variant !== "solid" && `is-${variant}`], title },
    iconName && icon(iconName, { size: 14 }), srPrefix && el("span", { class: "sr-only" }, srPrefix), el("span", {}, label));
}

export function accessBadge(access, { full = false } = {}) {
  const info = ACCESS[access] || ACCESS.unknown;
  return badge({ tone: info.tone, iconName: info.icon, label: full ? info.label : info.short, title: info.meaning });
}
export function originBadge(origin) {
  const info = ORIGIN[origin] || ORIGIN.unknown;
  return badge({ tone: info.tone, iconName: info.icon, label: info.label, variant: info.variant });
}
export function permissionBadge(permission) {
  const info = PERMISSION[permission] || PERMISSION.unknown;
  return badge({ tone: info.tone, iconName: info.icon, label: info.label, variant: info.variant });
}
export function kindBadge(kind) {
  const info = KINDS[kind] || KINDS.website;
  return badge({ tone: "dim", iconName: info.icon, label: info.label });
}
export function claimStatusBadge(status) {
  const info = CLAIM_STATUS[status] || CLAIM_STATUS.unreviewed;
  return badge({ tone: info.tone, iconName: info.icon, label: info.label, variant: info.variant });
}

// A 0–max bar. Null is "unknown": a hatched track with a dash, never an empty (zero) bar.
export function meter({ value, max = 100, label, text, compact = false }) {
  const known = isNumber(value);
  const ratio = known ? Math.min(1, Math.max(0, value / max)) : 0;
  return el("span", { class: ["meter", compact && "is-compact", !known && "is-unknown"], role: "img", "aria-label": label },
    el("span", { class: "meter-track", "aria-hidden": "true" },
      known && el("span", { class: "meter-fill", style: { width: `${(ratio * 100).toFixed(1)}%` } })),
    el("span", { class: "meter-value", "aria-hidden": "true" }, text ?? (known ? String(Math.round(value)) : "—")));
}

export const kindIcon = (kind) => icon((KINDS[kind] || KINDS.website).icon);

export function emptyState({ title, body, actions = [], tone = "" }) {
  return el("div", { class: ["empty", tone && `is-${tone}`] },
    el("h3", { class: "empty-title" }, title),
    body && el("p", { class: "empty-body" }, body),
    actions.length > 0 && el("div", { class: "empty-actions" }, actions));
}

export function skeletonRows(count = 6) {
  return el("div", { class: "skeleton-list", "aria-hidden": "true" },
    Array.from({ length: count }, () => el("div", { class: "skeleton-row" },
      el("span", { class: "skeleton-line is-title" }), el("span", { class: "skeleton-line" }), el("span", { class: "skeleton-line is-short" }))));
}
