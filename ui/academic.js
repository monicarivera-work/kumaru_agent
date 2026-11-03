/* Academic review: safe DOM text, explicit human decisions, no stored API keys. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  key: "", topics: [], sources: [], queue: false, offset: 0, pageSize: 30,
  selected: null, tab: "source", claims: [], claimId: null, dirty: false,
  sourceGeneration: 0, listGeneration: 0, editEvidenceId: null, editFindingId: null,
  pageLength: 0, claimOffset: 0, claimPageLength: 0, noticeOffset: 0,
};
const labels = {
  unknown: "Unknown", open: "Open page", account: "Account", paywall: "Paywall",
  inaccessible: "Inaccessible", human_verified: "Human verified", ai: "AI disclosed",
  sources_of_sources: "Sources of Sources", article: "Article", book: "Book", study: "Study", website: "Website",
};
const academicKinds = ["historical_acceptance", "credentials", "proof", "peer_review", "replication", "factual_criticism", "retraction", "corroboration"];
const pretty = (value) => labels[value] || value.replaceAll("_", " ");
const scoreText = (value) => value === null ? "Unknown" : `${value.toFixed(1)} / 100`;

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else if (name === "class") node.className = value;
    else if (["value", "checked", "selected", "disabled", "required", "multiple", "hidden"].includes(name)) node[name] = value;
    else node.setAttribute(name, String(value));
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function link(url, text = url) {
  const parsed = new URL(url);
  if (!["https:", "http:"].includes(parsed.protocol)) throw new Error("Source link is not HTTP(S).");
  return el("a", { href: parsed.href, target: "_blank", rel: "noopener noreferrer" }, text);
}

function input(name, value = "", type = "text", required = true) {
  return el("input", { name, value, type, required });
}
function area(name, value = "", required = true) {
  return el("textarea", { name, value, rows: 3, required });
}
function choice(name, choices, current, multiple = false) {
  const selected = Array.isArray(current) ? current : [current];
  return el("select", { name, multiple, ...(multiple ? { size: 4 } : {}) },
    choices.map(([value, text]) => el("option", { value, selected: selected.includes(value) }, text)));
}
function field(text, control, hint = "") {
  return el("label", { class: "form-field" }, el("span", {}, text), control, hint ? el("small", {}, hint) : null);
}
function checkbox(name, text, checked = false) {
  return el("label", { class: "check" }, el("input", { type: "checkbox", name, checked }), el("span", {}, text));
}
function grid(...fields) { return el("div", { class: "form-grid" }, fields); }
function topicChoices(ids = null) {
  return state.topics.filter((topic) => !ids || ids.includes(topic.id))
    .map((topic) => [topic.id, `${topic.name} (${topic.level})`]);
}
function feedback(message, error = false) {
  const node = $("feedback");
  node.replaceChildren(document.createTextNode(message));
  node.className = error ? "error" : "success";
  node.setAttribute("role", error ? "alert" : "status");
}
function discardChanges() {
  if (state.dirty && !window.confirm("Discard unsaved review changes?")) return false;
  state.dirty = false;
  return true;
}

async function api(path, body = undefined, method = "POST") {
  const response = await fetch(`/v1/academic${path}`, {
    method: body === undefined ? "GET" : method,
    credentials: "same-origin",
    headers: { ...(state.key ? { Authorization: `Bearer ${state.key}` } : {}), ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  const payload = await response.json();
  if (!response.ok) {
    if (response.status === 401) throw new Error("API key missing or incorrect. Enter the configured server key and reconnect.");
    if (response.status === 404) throw new Error("Academic API unavailable. Enable academic.enabled and restart Kumaru.");
    throw new Error(payload.error?.message || JSON.stringify(payload.detail || payload));
  }
  return payload;
}

async function action(button, task) {
  const original = button?.textContent;
  if (button) { button.disabled = true; button.textContent = "Working…"; }
  try { await task(); }
  catch (error) { feedback(error.message, true); }
  finally {
    if (button?.isConnected) {
      button.disabled = button.id === "previous-page" ? state.offset === 0
        : button.id === "next-page" ? state.pageLength < state.pageSize : false;
      button.textContent = original;
    }
  }
}

function reviewForm(children, submit, text = "Save review") {
  const button = el("button", { type: "submit" }, text);
  const form = el("form", { class: "review-form" }, children, button);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    action(button, async () => {
      await submit(new FormData(form));
      state.dirty = false;
      feedback("Review saved to the local catalogue.");
      await refreshSelected();
      document.querySelector(".tabs [aria-selected=true]")?.focus();
    });
  });
  return form;
}

async function loadSources() {
  const generation = ++state.listGeneration;
  const params = new URLSearchParams({ limit: state.pageSize, offset: state.offset });
  if (!state.queue) {
    if ($("topic-filter").value) params.set("topic_id", $("topic-filter").value);
    if ($("access-filter").value) params.set("access", $("access-filter").value);
  } else params.set("state", "reported");
  const items = await api(`${state.queue ? "/notices" : "/sources"}?${params}`);
  if (generation !== state.listGeneration) return;
  state.pageLength = items.length;
  if (!state.queue) state.sources = items;
  $("sources-heading").textContent = state.queue ? "Pending editorial notices" : "Sources";
  $("sources-button").hidden = !state.queue;
  $("page-label").textContent = `Page ${Math.floor(state.offset / state.pageSize) + 1}`;
  $("previous-page").disabled = state.offset === 0;
  $("next-page").disabled = items.length < state.pageSize;
  const content = $("source-list");
  if (!items.length) {
    content.replaceChildren(el("p", { class: "empty-copy" },
      state.queue ? "No pending notices on this page. A quiet queue is not proof that every source is current." : "No sources on this page. Discover academic metadata or add a known source by URL."));
    return;
  }
  const table = el("table", { class: "source-table" },
    el("thead", {}, el("tr", {}, el("th", { scope: "col" }, state.queue ? "Editorial notice" : "Source"), el("th", { scope: "col" }, state.queue ? "Type" : "Access"))));
  table.append(el("tbody", {}, items.map((item) => {
    const source = state.queue ? item.notice : item;
    const selected = state.selected?.source.id === source.source_id || state.selected?.source.id === source.id;
    const button = el("button", {
      type: "button", "aria-current": selected ? "true" : "false",
      onclick: () => action(null, async () => {
        if (!discardChanges()) return;
        if (state.queue) state.tab = "notices";
        await selectSource(state.queue ? source.source_id : source.id);
        $("review-workspace").focus({ preventScroll: true });
        if (window.matchMedia("(max-width: 760px)").matches) $("review-workspace").scrollIntoView({ block: "start" });
      }),
    }, source.title);
    return el("tr", { class: selected ? "selected" : "" },
      el("td", {}, button, el("small", {}, state.queue ? source.provider : `${pretty(source.kind)} · ${pretty(source.origin)}`)),
      el("td", {}, pretty(state.queue ? source.kind : source.access)));
  })));
  content.replaceChildren(table);
}

async function selectSource(id) {
  const generation = ++state.sourceGeneration;
  if (state.selected?.source.id !== id) {
    state.claimOffset = state.noticeOffset = 0;
    state.claimId = null;
  }
  $("source-detail").replaceChildren(el("p", { class: "empty-copy" }, "Loading source evidence…"));
  const [selected, claims, notices] = await Promise.all([
    api(`/sources/${id}`),
    api(`/claims?source_id=${id}&limit=30&offset=${state.claimOffset}`),
    api(`/notices?source_id=${id}&limit=30&offset=${state.noticeOffset}`),
  ]);
  if (generation !== state.sourceGeneration) return;
  state.claimPageLength = claims.length;
  if (state.claimId && !claims.some((item) => item.claim.id === state.claimId)) {
    const retained = await api(`/claims/${state.claimId}`);
    if (generation !== state.sourceGeneration) return;
    if (retained.claim.source_id === id) claims.push(retained);
  }
  selected.notices = notices;
  state.selected = selected;
  state.claims = claims;
  if (!claims.some((item) => item.claim.id === state.claimId)) state.claimId = claims[0]?.claim.id || null;
  state.editEvidenceId = state.editFindingId = null;
  renderDetail();
}
async function refreshSelected() {
  if (state.selected) await selectSource(state.selected.source.id);
  await loadSources();
}

function warnings(items) {
  return el("ul", { class: "warning-list" }, items.map((text) => el("li", {}, text)));
}
function scoreRow(topic, score) {
  return el("dl", { class: "score-row" },
    ...[
      ["Topic", state.topics.find((item) => item.id === topic)?.name || topic],
      ["Legitimacy", scoreText(score.legitimacy)], ["Usefulness", scoreText(score.usefulness)],
      ["Evidence coverage", `${Math.round(score.evidence_coverage * 100)}%`],
    ].map(([label, value]) => el("div", {}, el("dt", {}, label), el("dd", {}, value))));
}

function renderDetail() {
  const record = state.selected;
  const source = record.source;
  const heading = el("div", { class: "source-heading" },
    el("h2", { id: "review-heading" }, source.title),
    el("p", {}, link(source.url)),
    el("p", {}, `${pretty(source.kind)} · ${pretty(source.access)} · ${pretty(source.origin)}`),
    el("p", { class: "source-id" }, `Source ID: ${source.id}`));
  if (source.access_note) heading.append(el("p", {}, source.access_note));
  const pending = record.review_summary.pending_notices;
  const withdrawn = record.review_summary.withdrawn;
  const actions = el("div", { class: "source-actions" });
  for (const [label, endpoint, enabled] of [
    ["Inspect page", "inspect", true],
    ["Collect permitted text", "collect", source.kind !== "sources_of_sources" && source.origin === "human_verified" && source.permission === "text_permitted" && source.topic_reviewed && !pending && !withdrawn && ["unknown", "open"].includes(source.access)],
    ["Check editorial updates", "check-updates", Boolean(source.doi) && source.kind !== "sources_of_sources"],
  ]) {
    const button = el("button", { type: "button", class: "secondary", disabled: !enabled }, label);
    button.addEventListener("click", () => action(button, async () => {
      if (!discardChanges()) return;
      const result = await api(`/sources/${source.id}/${endpoint}`, {});
      feedback(endpoint === "check-updates" ? `Editorial check finished. ${result.warning}` : `${label} finished.`);
      await refreshSelected();
    }));
    actions.append(button);
  }
  heading.append(actions);
  if (pending || withdrawn) heading.append(el("p", { class: "status-label risk" }, "Editorial notice: trusted text use is blocked until review; confirmed retractions remain excluded."));
  const tabs = el("div", { class: "tabs", role: "tablist", "aria-label": "Source review sections" });
  for (const [id, label] of [["source", "Source evidence"], ["claims", `Claims (${record.review_summary.claim_count})`], ["notices", `Editorial notices (${record.review_summary.notice_count})`]]) {
    tabs.append(el("button", {
      type: "button", role: "tab", id: `tab-${id}`, "aria-selected": state.tab === id,
      "aria-controls": "tab-panel", tabindex: state.tab === id ? 0 : -1,
      onclick: () => { if (discardChanges()) { state.tab = id; renderDetail(); $(`tab-${id}`).focus(); } },
      onkeydown: (event) => {
        const ids = ["source", "claims", "notices"];
        if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
          event.preventDefault();
          if (!discardChanges()) return;
          const next = event.key === "Home" ? 0 : event.key === "End" ? 2 : (ids.indexOf(id) + (event.key === "ArrowRight" ? 1 : 2)) % 3;
          state.tab = ids[next]; renderDetail(); $(`tab-${state.tab}`).focus();
        }
      },
    }, label));
  }
  const panel = el("div", { id: "tab-panel", role: "tabpanel", "aria-labelledby": `tab-${state.tab}` });
  if (state.tab === "source") renderSourceReview(panel, source);
  else if (state.tab === "claims") renderClaims(panel, source);
  else renderNotices(panel, source);
  $("source-detail").replaceChildren(heading, tabs, panel);
}

function renderSourceReview(panel, source) {
  const scores = el("section", { class: "detail-section" },
    el("h3", {}, "Topic-scoped source scores"),
    el("p", {}, "Explainable estimates, not probabilities of truth. A source's proof does not automatically support every claim."));
  for (const score of state.selected.scores) {
    scores.append(scoreRow(score.topic_id, score), warnings(score.warnings),
      el("details", {}, el("summary", {}, "Scoring components and evidence IDs"), el("pre", {}, JSON.stringify(score, null, 2))));
  }
  panel.append(scores);
  const origin = choice("origin", [["unknown", "Unknown"], ["human_verified", "Human verified"], ["ai", "AI disclosed"]], source.origin);
  const permission = choice("permission", [["unknown", "Unknown"], ["metadata_only", "Metadata only"], ["text_permitted", "Text collection permitted"]], source.permission);
  const originUrl = input("origin_url", "", "url", false), originNote = area("origin_note", "", false);
  const rightsUrl = input("rights_url", "", "url", false), rightsNote = area("rights_note", "", false);
  function approvalRequirements() {
    originUrl.required = originNote.required = origin.value === "human_verified" && source.origin !== "human_verified";
    rightsUrl.required = rightsNote.required = permission.value === "text_permitted" && source.permission !== "text_permitted";
  }
  origin.addEventListener("change", approvalRequirements);
  permission.addEventListener("change", approvalRequirements);
  approvalRequirements();
  const form = reviewForm([
    grid(field("Human authorship", origin, "An AI detector cannot certify human authorship."),
      field("Copying permission", permission, "Free access or a subscription alone does not grant reuse rights.")),
    grid(field("Human-authorship evidence URL", originUrl), field("Why this establishes human provenance", originNote),
      field("Applicable license / permission URL", rightsUrl), field("Permission and collection scope", rightsNote)),
    field("Subjects covered", choice("topic_ids", topicChoices(), source.topic_ids, true), "Select one or more topics. Use Ctrl / Command to select multiple."),
    checkbox("topic_reviewed", "I reviewed the source's topic coverage.", source.topic_reviewed),
  ], async (data) => {
    const evidence = [];
    if (data.get("origin_url") && data.get("origin_note")) evidence.push({ kind: "origin", value: 1, url: data.get("origin_url"), note: data.get("origin_note"), reviewed: true });
    if (data.get("rights_url") && data.get("rights_note")) evidence.push({ kind: "permission", value: 1, url: data.get("rights_url"), note: data.get("rights_note"), reviewed: true });
    const body = { origin: data.get("origin"), permission: data.get("permission"), topic_reviewed: data.has("topic_reviewed"), evidence };
    if (data.has("topic_reviewed")) {
      const topics = data.getAll("topic_ids");
      if (!topics.length) throw new Error("Select at least one topic before confirming coverage.");
      body.topic_ids = topics;
    }
    await api(`/sources/${source.id}/review`, body);
  }, "Save provenance and topic review");
  panel.append(el("section", { class: "detail-section" }, el("h3", {}, "Provenance, rights and coverage"), form));
  const academicForm = reviewForm([
    grid(field("Evidence dimension", choice("kind", academicKinds.map((kind) => [kind, pretty(kind)]), "proof")),
      field("Assessed value (0–1)", el("input", { name: "value", type: "number", min: 0, max: 1, step: 0.01, value: 1, required: true }))),
    grid(field("Specific topic", choice("topic_id", topicChoices(source.topic_ids), source.topic_ids[0])),
      field("Evidence URL", input("url", "", "url"))),
    field("Finding and checked assumptions", area("note")),
    checkbox("reviewed", "I personally reviewed this finding, not merely its link."),
  ], async (data) => {
    await api(`/sources/${source.id}/review`, { evidence: [{
      ...(state.editEvidenceId ? { id: state.editEvidenceId } : {}),
      kind: data.get("kind"), value: Number(data.get("value")), url: data.get("url"),
      note: data.get("note"), topic_ids: [data.get("topic_id")], reviewed: data.has("reviewed"),
    }] });
    state.editEvidenceId = null;
  }, "Save academic finding");
  academicForm.id = "source-evidence-form";
  panel.append(el("section", { class: "detail-section" }, el("h3", {}, "Academic evidence"), academicForm));
  const existing = el("section", { class: "detail-section" }, el("h3", {}, "Recorded findings"));
  if (!source.evidence.length) existing.append(el("p", {}, "No evidence recorded yet. Discovery is not verification."));
  for (const item of source.evidence) {
    const row = findingRow(item);
    if (academicKinds.includes(item.kind)) row.querySelector(".finding-header").append(el("button", {
      type: "button", class: "secondary", onclick: () => {
        if (!discardChanges()) return;
        state.editEvidenceId = item.id;
        for (const key of ["kind", "value", "url", "note"]) academicForm.elements.namedItem(key).value = item[key];
        academicForm.elements.namedItem("topic_id").value = item.topic_ids[0] || source.topic_ids[0];
        academicForm.elements.namedItem("reviewed").checked = item.reviewed;
        academicForm.elements.namedItem("note").focus();
      },
    }, "Edit finding"));
    existing.append(row);
  }
  panel.append(existing);
  panel.append(el("details", { class: "inline-details" },
    el("summary", {}, "Source and editorial review history"),
    el("pre", {}, JSON.stringify(state.selected.reviews, null, 2))));
  const graph = state.selected.graph;
  panel.append(el("section", { class: "detail-section" }, el("h3", {}, "Source relationships"),
    el("ul", { class: "relationship-list" },
      ...["ancestors", "descendants", "siblings"].map((kind) => el("li", {}, `${pretty(kind)}: ${graph[kind].join(", ") || "None recorded"}`)),
      ...graph.links.map((edge) => el("li", {}, `${pretty(edge.kind)}: ${edge.source_id} → ${edge.target_id}. ${edge.note}`)))));
}

function findingRow(item) {
  return el("article", { class: "finding" },
    el("div", { class: "finding-header" }, el("strong", {}, `${pretty(item.kind)} · ${item.value.toFixed(2)}`)),
    el("small", {}, `${item.reviewed ? "Reviewed" : "Unreviewed"}${item.stance ? ` · ${pretty(item.stance)}` : ""}${item.locator ? ` · ${item.locator}` : ""}`),
    el("p", {}, item.note), el("p", {}, link(item.url)), el("small", {}, `Evidence ID: ${item.id}`));
}

function renderClaims(panel, source) {
  if (source.kind === "sources_of_sources") {
    panel.append(el("p", { class: "empty-copy" }, "Record claims on individual child works, not on a journal or library catalogue."));
    return;
  }
  const createForm = reviewForm([
    field("Assertion or theorem", area("statement")),
    grid(field("Topic", choice("topic_id", topicChoices(source.topic_ids), source.topic_ids[0])),
      field("Exact location", input("locator"), "Page, section, theorem or stable anchor.")),
    field("Assumptions (one per line)", area("assumptions", "", false)),
  ], async (data) => {
    const result = await api("/claims", {
      source_id: source.id, topic_id: data.get("topic_id"), statement: data.get("statement"),
      locator: data.get("locator"), assumptions: data.get("assumptions").split("\n").map((value) => value.trim()).filter(Boolean),
    });
    state.claimId = result.claim.id;
  }, "Record unreviewed claim");
  panel.append(el("details", { class: "inline-details" }, el("summary", {}, "Record an individual claim"), createForm));
  panel.append(detailPagination("claims", state.claimOffset, state.selected.review_summary.claim_count));
  if (!state.claims.length) {
    panel.append(el("p", { class: "empty-copy" }, "No claims recorded. Capture an assertion, its exact location and assumptions before attaching proof or criticism."));
    return;
  }
  const picker = choice("claim-picker", state.claims.map((item) => [item.claim.id, item.claim.statement.slice(0, 100)]), state.claimId);
  picker.addEventListener("change", () => {
    if (!discardChanges()) { picker.value = state.claimId; return; }
    state.claimId = picker.value; state.editFindingId = null; renderDetail();
  });
  panel.append(field("Select a claim", picker));
  const current = state.claims.find((item) => item.claim.id === state.claimId);
  const claim = current.claim;
  panel.append(el("section", { class: "detail-section" }, el("h3", {}, claim.statement),
    el("p", {}, `Location: ${claim.locator}`),
    el("p", { class: "status-label" }, `Assessment: ${pretty(current.assessment.status)}. This is not a guarantee of truth.`),
    el("ul", { class: "assumptions" }, claim.assumptions.map((item) => el("li", {}, item))),
    scoreRow(claim.topic_id, current.assessment.score), warnings(current.assessment.score.warnings)));
  panel.append(el("section", { class: "detail-section" }, el("h3", {}, "Review claim attribution"),
    reviewForm([
      checkbox("reviewed", "I checked this claim's attribution, topic, location and assumptions.", claim.reviewed),
      field("Review explanation", area("note")),
    ], async (data) => {
      await api(`/claims/${claim.id}/review`, { reviewed: data.has("reviewed"), note: data.get("note") });
    })));
  const form = reviewForm([
    grid(field("Referenced source ID", input("source_id", source.id), "Use an existing catalogue source ID; its human provenance and topic coverage must be reviewed."),
      field("Relationship to the claim", choice("stance", [["supports", "Supports"], ["challenges", "Challenges"], ["context", "Context only"]], "supports"))),
    grid(field("Evidence dimension", choice("kind", academicKinds.map((kind) => [kind, pretty(kind)]), "proof")),
      field("Assessed strength / flaw severity (0–1)", el("input", { name: "value", type: "number", min: 0, max: 1, step: 0.01, value: 1, required: true }))),
    grid(field("Evidence URL", input("url", "", "url")), field("Exact evidence location", input("locator"))),
    field("Proof, assumptions or specific factual flaw", area("note")),
    checkbox("reviewed", "I reviewed this evidence and its applicability to this exact claim."),
  ], async (data) => {
    await api(`/claims/${claim.id}/review`, {
      reviewed: claim.reviewed, note: data.get("note"), evidence: [{
        ...(state.editFindingId ? { id: state.editFindingId } : {}),
        source_id: data.get("source_id"), stance: data.get("stance"), kind: data.get("kind"),
        value: Number(data.get("value")), url: data.get("url"), locator: data.get("locator"),
        note: data.get("note"), topic_ids: [claim.topic_id], reviewed: data.has("reviewed"),
      }],
    });
    state.editFindingId = null;
  }, "Save claim finding");
  form.elements.namedItem("stance").addEventListener("change", (event) => {
    const kind = form.elements.namedItem("kind");
    if (event.target.value === "challenges") kind.value = "factual_criticism";
    else if (["factual_criticism", "retraction"].includes(kind.value)) kind.value = "proof";
  });
  panel.append(el("section", { class: "detail-section" }, el("h3", {}, "Attach claim-specific evidence"), form));
  const findings = el("section", { class: "detail-section" }, el("h3", {}, "Supporting and challenging findings"));
  for (const item of claim.evidence) {
    const row = findingRow(item);
    row.querySelector(".finding-header").append(el("button", {
      type: "button", class: "secondary", onclick: () => {
        if (!discardChanges()) return;
        state.editFindingId = item.id;
        for (const key of ["source_id", "stance", "kind", "value", "url", "locator", "note"]) form.elements.namedItem(key).value = item[key];
        form.elements.namedItem("reviewed").checked = item.reviewed;
        form.elements.namedItem("note").focus();
      },
    }, "Edit finding"));
    findings.append(row);
  }
  if (!claim.evidence.length) findings.append(el("p", {}, "No claim-specific findings. Whole-source evidence is not inherited."));
  findings.append(el("details", {}, el("summary", {}, "Claim review history"), el("pre", {}, JSON.stringify(current.reviews, null, 2))));
  panel.append(findings);
}

function renderNotices(panel, source) {
  const check = state.selected.update_check;
  panel.append(el("p", { class: "empty-copy" }, check.checked_at
    ? `Last check: ${check.checked_at}.${check.error ? ` Failed: ${check.error}` : check.partial ? " Partial scan; additional records may exist." : " Absence of a notice does not prove that a source is current."}`
    : "No editorial check recorded. Crossref monitoring requires a DOI; you can also report a notice manually."));
  const reportForm = reviewForm([
    grid(field("Notice title", input("title")),
      field("Notice type", choice("kind", [["correction", "Correction"], ["retraction", "Retraction"], ["expression_of_concern", "Expression of concern"], ["update", "Other editorial update"]], "correction"))),
    grid(field("Official notice URL", input("url", "", "url")), field("Event date (optional)", input("event_date", "", "date", false))),
  ], async (data) => {
    await api("/notices", { source_id: source.id, title: data.get("title"), kind: data.get("kind"), url: data.get("url"), event_date: data.get("event_date") || null });
  }, "Report notice for review");
  panel.append(el("details", { class: "inline-details" }, el("summary", {}, "Report a manually found notice"), reportForm));
  panel.append(detailPagination("notices", state.noticeOffset, state.selected.review_summary.notice_count));
  if (!state.selected.notices.length) panel.append(el("p", { class: "empty-copy" }, "No editorial notices recorded for this source."));
  for (const item of state.selected.notices) {
    const notice = item.notice;
    const select = choice("state", [["reported", "Reported / needs review"], ["confirmed", "Confirmed by review"], ["dismissed", "Dismissed with explanation"]], notice.state);
    const impact = el("input", { name: "impact", type: "number", min: 0, max: 1, step: 0.01, value: notice.impact ?? "", required: notice.state === "confirmed" });
    select.addEventListener("change", () => { impact.required = select.value === "confirmed"; });
    const form = reviewForm([
      grid(field("Review decision", select), field("Assessed factual impact (0–1)", impact, "A confirmed retraction requires 1. Zero can acknowledge a non-factual correction.")),
      field("Affected topics", choice("affected_topic_ids", topicChoices(source.topic_ids), notice.affected_topic_ids, true), "Required for non-retraction factual impacts above zero."),
      field("What you verified / why you dismissed it", area("note", notice.review_note)),
      el("p", { class: "help" }, "Confirming clears cached text and can change scores. Reported notices block trusted text use until reviewed."),
    ], async (data) => {
      await api(`/notices/${notice.id}/review`, {
        state: data.get("state"), impact: data.get("impact") === "" ? null : Number(data.get("impact")),
        affected_topic_ids: data.getAll("affected_topic_ids"), note: data.get("note"),
      });
    }, "Save notice review");
    const block = el("article", { class: "notice" },
      el("h3", {}, notice.title),
      el("p", { class: `status-label ${notice.state === "reported" ? "risk" : ""}` }, `${pretty(notice.kind)} · ${notice.state} · ${notice.provider}`),
      el("p", {}, link(notice.url)),
      el("p", {}, `Affected claims: ${item.affected_claim_ids.length}. Dependent references: ${item.dependent_source_ids.length}.`),
      el("details", {}, el("summary", {}, "Affected claim and source IDs"), el("pre", {}, JSON.stringify({
        claim_ids: item.affected_claim_ids, source_ids: item.dependent_source_ids,
      }, null, 2))), form);
    if (item.affected_claim_ids.length) block.insertBefore(el("details", {},
      el("summary", {}, "Open affected claims"),
      ...item.affected_claim_ids.map((id) => el("p", {}, el("button", {
        type: "button", class: "secondary", onclick: () => action(null, async () => {
          if (!discardChanges()) return;
          const result = await api(`/claims/${id}`);
          if (state.selected.source.id !== result.claim.source_id) await selectSource(result.claim.source_id);
          state.claimId = id; state.tab = "claims"; await refreshSelected();
        }),
      }, `Review claim ${id}`)))), form);
    panel.append(block);
  }
}

function detailPagination(kind, offset, total) {
  const pager = el("div", { class: "pagination", "aria-label": `${pretty(kind)} pages` });
  for (const [label, delta, disabled] of [
    ["Previous", -1, offset === 0],
    ["Next", 1, offset + 30 >= total],
  ]) {
    const button = el("button", { type: "button", class: "secondary", disabled }, label);
    button.addEventListener("click", () => action(button, async () => {
      if (!discardChanges()) return;
      if (kind === "claims") { state.claimOffset = Math.max(0, state.claimOffset + delta * 30); state.claimId = null; }
      else state.noticeOffset = Math.max(0, state.noticeOffset + delta * 30);
      await refreshSelected();
    }));
    if (delta === 1) pager.append(el("span", {}, `Page ${Math.floor(offset / 30) + 1} · ${total} ${kind}`));
    pager.append(button);
  }
  return pager;
}

$("connection-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action(event.submitter, async () => {
    if (!discardChanges()) return;
    state.key = $("api-key").value;
    await initialize();
  });
});
$("discovery-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action(event.submitter, async () => {
    if (!discardChanges()) return;
    const result = await api("/search", { topic_id: $("topic-filter").value, query: $("discovery-query").value, provider: $("search-provider").value, limit: 20 });
    state.queue = false; state.offset = 0;
    await loadSources();
    if (result.source_ids[0]) await selectSource(result.source_ids[0]);
    feedback(`Discovered ${result.source_ids.length} candidate sources. Authorship, accuracy and copying rights still require review.`);
  });
});
$("add-source-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action(event.submitter, async () => {
    if (!discardChanges()) return;
    const data = new FormData(event.target);
    const source = await api("/sources", {
      url: data.get("url"), title: data.get("title"), kind: data.get("kind"),
      doi: data.get("doi") || null, topic_ids: [$("topic-filter").value],
    });
    state.queue = false; state.offset = 0;
    await loadSources(); await selectSource(source.id); event.target.reset();
    feedback("Source candidate catalogued. Review provenance and topic coverage before relying on it.");
  });
});
for (const id of ["topic-filter", "access-filter"]) $(id).addEventListener("change", () => action(null, async () => {
  state.offset = 0; await loadSources();
}));
for (const [id, delta] of [["previous-page", -1], ["next-page", 1]]) $(id).addEventListener("click", () => action($(id), async () => {
  state.offset = Math.max(0, state.offset + delta * state.pageSize); await loadSources();
}));
$("queue-button").addEventListener("click", () => action($("queue-button"), async () => {
  state.queue = true; state.offset = 0; await loadSources();
}));
$("sources-button").addEventListener("click", () => action($("sources-button"), async () => {
  state.queue = false; state.offset = 0; await loadSources();
}));
$("monitor-button").addEventListener("click", () => action($("monitor-button"), async () => {
  if (!discardChanges()) return;
  const result = await api("/monitor?limit=10", {});
  await refreshSelected();
  feedback(`Editorial checks: ${result.results.length} completed, ${result.errors.length} failed. New notices are reports, not confirmed findings.`, result.errors.length > 0);
  if (result.errors.length) $("feedback").append(el("details", {}, el("summary", {}, "Check failures"), el("pre", {}, JSON.stringify(result.errors, null, 2))));
}));
document.addEventListener("input", (event) => {
  if (event.target.closest("#tab-panel form")) state.dirty = true;
});
window.addEventListener("beforeunload", (event) => {
  if (state.dirty) { event.preventDefault(); event.returnValue = ""; }
});

async function initialize() {
  const [topics, monitor] = await Promise.all([api("/topics"), api("/monitor/status")]);
  state.topics = topics;
  const selected = $("topic-filter").value || "mathematics";
  $("topic-filter").replaceChildren(...topicChoices().map(([id, name]) => el("option", { value: id, selected: id === selected }, name)));
  await loadSources();
  if (state.selected) await selectSource(state.selected.source.id);
  feedback(monitor.periodic_enabled
    ? `Connected. Periodic editorial checks run every ${monitor.interval_s} seconds in batches of ${monitor.batch_size}.`
    : "Connected. Periodic monitoring is off; use the editorial check controls when needed.");
}
action(null, initialize);
