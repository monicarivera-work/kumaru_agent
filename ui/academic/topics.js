// Subjects: the topic index, the ARIA tree in the left rail, the topic dialog and the
// checkbox tree other dialogs use to tag sources.
import { api } from "./api.js";
import { applyPendingFocus, openDialog, toast } from "./dialogs.js";
import { button, clear, el, fmtInt, levelLabel, LEVELS, uid } from "./dom.js";
import { areaField, textField } from "./forms.js";
import { icon } from "./icons.js";
import { emit, on, state } from "./state.js";

let byId = new Map();
let children = new Map();

export async function loadTopics() {
  state.topics = await api("/topics");
  byId = new Map(state.topics.map((topic) => [topic.id, topic]));
  children = new Map();
  for (const topic of state.topics) {
    if (!topic.parent_id) continue;
    children.set(topic.parent_id, [...(children.get(topic.parent_id) || []), topic]);
  }
  for (const list of children.values()) list.sort((a, b) => a.name.localeCompare(b.name));
  if (!byId.has(state.topicId)) state.topicId = state.topics.find((topic) => !topic.parent_id)?.id || "";
  if (state.expanded.size === 0) {
    const start = pathOf(state.topicId)[0];
    if (start) {
      state.expanded.add(start.id);
      childrenOf(start.id).forEach((child) => state.expanded.add(child.id));
    }
  }
  emit("topics-changed");
}

export const topicById = (id) => byId.get(id);
export const topicName = (id) => byId.get(id)?.name || id;
export const childrenOf = (id) => children.get(id) || [];
export const rootTopics = () => state.topics.filter((topic) => !topic.parent_id).sort((a, b) => a.name.localeCompare(b.name));

export function pathOf(id) {
  const path = [];
  for (let topic = byId.get(id); topic; topic = topic.parent_id ? byId.get(topic.parent_id) : null) path.unshift(topic);
  return path;
}

export function descendantIds(id) {
  const found = new Set();
  const walk = (topicId) => { found.add(topicId); childrenOf(topicId).forEach((child) => walk(child.id)); };
  if (byId.has(id)) walk(id);
  return found;
}

export function breadcrumb(id) {
  const path = pathOf(id);
  return el("span", { class: "crumbs" }, path.map((topic, index) => [
    index > 0 && el("span", { class: "crumb-sep", "aria-hidden": "true" }, "›"),
    el("span", { class: index === path.length - 1 ? "crumb is-current" : "crumb" }, topic.name),
  ]));
}
export const breadcrumbText = (id) => pathOf(id).map((topic) => topic.name).join(" › ");

function rolledCounts() {
  const source = state.overview || state.queue || {};
  // The server counts each source once per subtree; summing direct tags would double-count
  // a source tagged with several topics, so that is only the fallback.
  if (source.topic_totals) return new Map(Object.entries(source.topic_totals));
  const direct = source.topic_counts || {};
  const totals = new Map();
  const total = (id) => {
    if (!totals.has(id)) totals.set(id, (direct[id] || 0) + childrenOf(id).reduce((sum, child) => sum + total(child.id), 0));
    return totals.get(id);
  };
  state.topics.forEach((topic) => total(topic.id));
  return totals;
}

export function ensureVisible(id) {
  for (const topic of pathOf(id).slice(0, -1)) state.expanded.add(topic.id);
}

export function selectTopic(id) {
  if (!byId.has(id) || id === state.topicId) return;
  state.topicId = id;
  ensureVisible(id);
  emit("topic-selected", { id });
}

export function topicOptions({ placeholder = "" } = {}) {
  const options = placeholder ? [{ value: "", label: placeholder }] : [];
  const walk = (topic, depth) => {
    options.push({ value: topic.id, label: `${"\u2003".repeat(depth)}${topic.name} · ${levelLabel(topic.level)}` });
    childrenOf(topic.id).forEach((child) => walk(child, depth + 1));
  };
  rootTopics().forEach((topic) => walk(topic, 0));
  return options;
}

/* ─── Subjects tree ─── */

function treeNode(topic, level, totals, controls) {
  const kids = childrenOf(topic.id);
  const expanded = state.expanded.has(topic.id);
  const selected = state.topicId === topic.id;
  const count = totals.get(topic.id) || 0;
  const row = el("div", { class: "tree-row" },
    el("span", { class: ["tree-twist", expanded && "is-open", kids.length === 0 && "is-leaf"], "aria-hidden": "true", dataset: { twist: kids.length ? "1" : "" } },
      kids.length > 0 && icon("chevron", { size: 14 })),
    el("span", { class: "tree-name" }, topic.name),
    el("span", { class: "tree-level" }, levelLabel(topic.level)),
    el("span", { class: "tree-count", title: "Sources tagged with this subject or anything under it, each counted once." }, fmtInt(count)));
  return el("li", {
    role: "treeitem", class: ["tree-item", selected && "is-selected"], tabindex: selected ? 0 : -1, "aria-level": level,
    "aria-selected": selected, "aria-expanded": kids.length ? expanded : undefined,
    "aria-label": `${topic.name}, ${levelLabel(topic.level)}, ${fmtInt(count)} ${count === 1 ? "source" : "sources"}`,
    dataset: { topicId: topic.id },
  }, row, kids.length > 0 && el("ul", { role: "group", hidden: !expanded }, kids.map((child) => treeNode(child, level + 1, totals, controls))));
}

function visibleItems(tree) {
  return [...tree.querySelectorAll('[role="treeitem"]')].filter((item) => !item.parentElement.closest('[role="group"][hidden]'));
}

function wireTree(tree, rerender) {
  const focusItem = (item) => {
    if (!item) return;
    tree.querySelectorAll('[role="treeitem"][tabindex="0"]').forEach((node) => node.setAttribute("tabindex", "-1"));
    item.setAttribute("tabindex", "0");
    item.focus();
  };
  const toggle = (id, open) => {
    if (open) state.expanded.add(id); else state.expanded.delete(id);
    rerender(id);
  };
  tree.addEventListener("click", (event) => {
    const row = event.target.closest(".tree-row");
    const item = event.target.closest('[role="treeitem"]');
    if (!row || !item) return;
    const id = item.dataset.topicId;
    if (event.target.closest("[data-twist]")?.dataset.twist) { toggle(id, !state.expanded.has(id)); return; }
    if (childrenOf(id).length && !state.expanded.has(id) && id !== state.topicId) state.expanded.add(id);
    selectTopic(id);
    rerender(id);
  });
  tree.addEventListener("keydown", (event) => {
    const item = event.target.closest('[role="treeitem"]');
    if (!item || event.altKey || event.ctrlKey || event.metaKey) return;
    const id = item.dataset.topicId;
    const items = visibleItems(tree);
    const index = items.indexOf(item);
    const hasKids = childrenOf(id).length > 0;
    const keys = {
      ArrowDown: () => focusItem(items[index + 1]),
      ArrowUp: () => focusItem(items[index - 1]),
      Home: () => focusItem(items[0]),
      End: () => focusItem(items.at(-1)),
      ArrowRight: () => {
        if (!hasKids) return;
        if (!state.expanded.has(id)) toggle(id, true); else focusItem(item.querySelector('[role="treeitem"]'));
      },
      ArrowLeft: () => {
        if (hasKids && state.expanded.has(id)) toggle(id, false);
        else focusItem(item.parentElement.closest('[role="treeitem"]'));
      },
      Enter: () => { selectTopic(id); rerender(id); },
      " ": () => { selectTopic(id); rerender(id); },
    };
    if (!keys[event.key]) return;
    event.preventDefault();
    keys[event.key]();
  });
}

export function renderSubjects(host) {
  const tree = el("ul", { role: "tree", class: "tree", "aria-label": "Subjects" });
  const empty = el("p", { class: "quiet tree-empty" }, "Subjects load once Kumaru is connected.");
  const actions = el("div", { class: "rail-actions" });
  const build = (focusId = null) => {
    const active = tree.contains(document.activeElement) ? document.activeElement.closest('[role="treeitem"]')?.dataset.topicId : null;
    const totals = rolledCounts();
    clear(tree, rootTopics().map((topic) => treeNode(topic, 1, totals)));
    empty.hidden = state.topics.length > 0;
    const current = state.topicId;
    if (!tree.querySelector('[tabindex="0"]')) tree.querySelector('[role="treeitem"]')?.setAttribute("tabindex", "0");
    const keep = focusId || active;
    if (keep) {
      const target = tree.querySelector(`[data-topic-id="${CSS.escape(keep)}"]`) || tree.querySelector(`[data-topic-id="${CSS.escape(current)}"]`);
      tree.querySelectorAll('[role="treeitem"][tabindex="0"]').forEach((node) => node.setAttribute("tabindex", "-1"));
      target?.setAttribute("tabindex", "0");
      target?.focus();
    }
    const selected = state.topics.find((topic) => topic.id === current);
    const deepest = LEVELS.findIndex((level) => level.id === selected?.level) >= LEVELS.length - 1;
    clear(actions,
      button("Add topic", { variant: "secondary", iconName: "plus", disabled: !selected || deepest, focusId: "add-topic",
        title: deepest ? "Sub-chapters are the deepest level" : `Add a topic under ${selected?.name || "the selected subject"}`,
        onclick: (event) => openTopicDialog({ mode: "child", trigger: event.currentTarget }) }),
      button("Add area", { variant: "secondary", iconName: "plus", focusId: "add-area", onclick: (event) => openTopicDialog({ mode: "area", trigger: event.currentTarget }) }),
      button("Edit topic", { variant: "quiet", iconName: "edit", disabled: !selected, focusId: "edit-topic", onclick: (event) => openTopicDialog({ mode: "edit", trigger: event.currentTarget }) }));
    applyPendingFocus();
  };
  wireTree(tree, build);
  clear(host, el("h2", { class: "rail-title", id: "subjects-title" }, "Subjects"), tree, empty, actions);
  build();
  on("topics-changed", () => build());
  on("overview-changed", () => build());
  on("topic-selected", () => build());
}

/* ─── Topic dialog ─── */

const slugify = (value) => value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase()
  .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 100);

export async function openTopicDialog({ mode, trigger }) {
  const selected = byId.get(state.topicId);
  const editing = mode === "edit";
  let parent = null;
  let level = "area";
  if (mode === "child") {
    parent = selected;
    level = LEVELS[LEVELS.findIndex((item) => item.id === selected.level) + 1]?.id;
  }
  if (editing) { parent = selected.parent_id ? byId.get(selected.parent_id) : null; level = selected.level; }
  if (!level) return;
  const title = editing ? `Edit ${selected.name}` : mode === "area" ? "Add area" : `Add ${levelLabel(level).toLowerCase()}`;
  const placement = parent ? `Placed under ${parent.name}.` : "A new root of the subject tree.";
  const result = await openDialog({
    title, trigger, size: "md",
    description: editing ? "Rename the topic or change its aliases. Its ID, level and place in the tree stay the same."
      : `${levelLabel(level)} · ${placement} Topics tag sources; they never approve them.`,
    submitLabel: editing ? "Save topic" : "Add topic",
    build: () => {
      const name = textField({ name: "name", label: "Name", value: editing ? selected.name : "", required: true, maxlength: 200 });
      const id = textField({
        name: "id", label: "Topic ID", value: editing ? selected.id : "", required: true, maxlength: 100,
        pattern: "[a-z0-9][a-z0-9._\\-]*", readOnly: editing,
        hint: editing ? "IDs cannot change because sources and claims refer to them." : "Lowercase letters, digits, dots, hyphens and underscores. Filled in from the name until you edit it.",
      });
      let custom = editing;
      id.control.addEventListener("input", () => { custom = true; });
      name.control.addEventListener("input", () => { if (!custom) id.control.value = slugify(name.control.value); });
      const aliases = areaField({
        name: "aliases", label: "Aliases", rows: 3, value: editing ? selected.aliases.join("\n") : "",
        hint: "Other names for this topic, one per line. They are recorded for taxonomy consumers; searches use the topic name.",
      });
      return [name.node, id.node, aliases.node];
    },
    onSubmit: async (data) => {
      const id = String(data.get("id")).trim();
      const aliases = String(data.get("aliases")).split("\n").map((item) => item.trim()).filter(Boolean);
      if (!editing && byId.has(id)) {
        throw new Error(`A topic with the ID “${id}” already exists. Choose a different ID, or select it and use Edit topic.`);
      }
      const saved = await api(`/topics/${encodeURIComponent(id)}`, {
        method: "PUT", body: { id, name: String(data.get("name")).trim(), level, parent_id: parent?.id || null, aliases },
      });
      return saved;
    },
  });
  if (!result?.id) return;
  if (parent) state.expanded.add(parent.id);
  await loadTopics();
  if (!editing) selectTopic(result.id);
  emit("data-changed", { scope: "topics" });
  toast({ message: editing ? `Saved ${result.name}.` : `Added ${result.name} to the subject tree.` });
}

/* ─── Topic checkbox tree ─── */

// Nested checkboxes named `name`; selecting a parent does not select its children.
export function topicCheckboxTree({ name = "topic_ids", selected = [], legend = "Subjects", hint = "" } = {}) {
  const chosen = new Set(selected);
  const hintId = hint ? uid("hint") : "";
  const branch = (topic) => {
    const id = uid("topic");
    const input = el("input", { id, type: "checkbox", name, value: topic.id, checked: chosen.has(topic.id), class: "check-input" });
    const kids = childrenOf(topic.id);
    return el("li", {}, el("label", { class: "checkbox", for: id }, input,
      el("span", { class: "check-text" }, topic.name, el("span", { class: "check-level" }, levelLabel(topic.level)))),
    kids.length > 0 && el("ul", {}, kids.map(branch)));
  };
  const node = el("fieldset", { class: "field topic-tree-field", "aria-describedby": hintId || undefined },
    el("legend", { class: "field-label" }, legend),
    hint && el("p", { class: "field-hint", id: hintId }, hint),
    el("ul", { class: "topic-checks" }, rootTopics().map(branch)));
  return { node, values: () => [...node.querySelectorAll("input:checked")].map((input) => input.value) };
}
