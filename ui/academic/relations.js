// Relations tab: the containment family (ancestors → descendants, siblings), the other links
// grouped by direction, and the dialog that adds a relationship in plain sentences.
import { api } from "./api.js";
import { openDialog, toast } from "./dialogs.js";
import { accessBadge, button, el, emptyState, kindIcon, plural } from "./dom.js";
import { areaField, selectField } from "./forms.js";
import { sourcePicker } from "./picker.js";
import { emit, navigate, sourceHash } from "./state.js";

const SHOWN = 6;

const SENTENCES = [
  { value: "contains", label: "This source contains…", kind: "contains", reverse: false,
    hint: "Membership is not peer review, and it does not give Kumaru the full text." },
  { value: "contained", label: "This source is contained in…", kind: "contains", reverse: true,
    hint: "The other source becomes a parent; its own parents become this source’s ancestors." },
  { value: "cites", label: "This source cites…", kind: "cites", reverse: false, hint: "A citation is context, not evidence that either source is correct." },
  { value: "cited", label: "This source is cited by…", kind: "cites", reverse: true, hint: "A citation is context, not evidence that either source is correct." },
  { value: "corroborates", label: "This source corroborates…", kind: "corroborates", reverse: false,
    hint: "Counts toward the other source’s independent corroboration only if both are human verified for the same reviewed subject, with different publishers and no shared authors." },
  { value: "corroborated", label: "This source is corroborated by…", kind: "corroborates", reverse: true,
    hint: "Counts toward this source’s independent corroboration only if both are human verified for the same reviewed subject, with different publishers and no shared authors." },
  { value: "criticizes", label: "This source criticizes…", kind: "criticizes", reverse: false,
    hint: "A criticism link alone does not change a score. Record a reviewed finding for that." },
  { value: "criticized", label: "This source is criticized by…", kind: "criticizes", reverse: true,
    hint: "A criticism link alone does not change a score. Record a reviewed finding for that." },
  { value: "open_copy", label: "This source has a free copy at…", kind: "open_copy", reverse: false,
    hint: "A free copy can be a different version; check it before relying on it." },
  { value: "copy_of", label: "This source is a free copy of…", kind: "open_copy", reverse: true,
    hint: "A free copy can be a different version; check it before relying on it." },
  { value: "related", label: "This source is related to…", kind: "related", reverse: false, hint: "Use this when none of the other relationships fit." },
];

const unavailable = (id) => ({ id, title: "Source no longer in the catalogue", kind: "website", access: "unknown", origin: "unknown" });

function nodeButton(card) {
  return el("button", { type: "button", class: "node", onclick: () => navigate(sourceHash(card.id, "relations")) },
    kindIcon(card.kind), el("span", { class: "node-title" }, card.title), accessBadge(card.access));
}

function nodeList(ids, cards) {
  const list = el("ul", { class: "node-list" });
  let expanded = false;
  const render = () => {
    list.replaceChildren(...ids.slice(0, expanded ? ids.length : SHOWN).map((id) => el("li", {}, nodeButton(cards[id] || unavailable(id)))));
    if (ids.length > SHOWN) {
      list.append(el("li", {}, button(expanded ? "Show fewer" : `Show ${ids.length - SHOWN} more`, { variant: "quiet", onclick: () => { expanded = !expanded; render(); } })));
    }
  };
  render();
  return list;
}

function column(title, ids, cards, empty) {
  return el("section", { class: "family-col" },
    el("h4", { class: "family-title" }, title, el("span", { class: "family-count" }, ids.length || "")),
    ids.length ? nodeList(ids, cards) : el("p", { class: "quiet" }, empty));
}

function linkGroups(ctx) {
  const id = ctx.id;
  const { links } = ctx.detail.graph;
  const groups = [
    ["Cites", (link) => link.kind === "cites" && link.source_id === id, (link) => link.target_id],
    ["Cited by", (link) => link.kind === "cites" && link.target_id === id, (link) => link.source_id],
    ["Corroborates", (link) => link.kind === "corroborates" && link.source_id === id, (link) => link.target_id],
    ["Corroborated by", (link) => link.kind === "corroborates" && link.target_id === id, (link) => link.source_id],
    ["Criticizes", (link) => link.kind === "criticizes" && link.source_id === id, (link) => link.target_id],
    ["Criticized by", (link) => link.kind === "criticizes" && link.target_id === id, (link) => link.source_id],
    ["Free copies", (link) => link.kind === "open_copy" && link.source_id === id, (link) => link.target_id],
    ["Free copy of", (link) => link.kind === "open_copy" && link.target_id === id, (link) => link.source_id],
    ["Related", (link) => link.kind === "related", (link) => (link.source_id === id ? link.target_id : link.source_id)],
  ];
  return groups.map(([title, match, pick]) => [title, links.filter(match).map((link) => ({ card: ctx.detail.related[pick(link)] || unavailable(pick(link)), note: link.note }))])
    .filter(([, items]) => items.length);
}

export function renderRelationsTab(ctx) {
  const { graph, related } = ctx.detail;
  const parents = graph.links.filter((link) => link.kind === "contains" && link.target_id === ctx.id).map((link) => link.source_id);
  const children = graph.links.filter((link) => link.kind === "contains" && link.source_id === ctx.id).map((link) => link.target_id);
  const ancestors = graph.ancestors.filter((id) => !parents.includes(id));
  const descendants = graph.descendants.filter((id) => !children.includes(id));
  const others = linkGroups(ctx);
  const nothing = !graph.links.length && !graph.siblings.length;
  return el("div", { class: "tab-body" },
    el("div", { class: "tab-head" },
      el("h3", { class: "tab-title" }, "Relations"),
      button("Add relationship", { variant: "primary", iconName: "plus", focusId: "add-relationship", onclick: (event) => openRelationshipDialog({ source: ctx.source, trigger: event.currentTarget }) })),
    nothing && emptyState({ title: "No relationships recorded", body: "Link this source to the container that lists it, to sources it cites, to a free copy of it, or to independent sources that corroborate it." }),
    !nothing && el("div", { class: "family", role: "group", "aria-label": "Containment family" },
      column("Ancestors", ancestors, related, "None"),
      column("Parents", parents, related, "None"),
      el("section", { class: "family-col is-self" },
        el("h4", { class: "family-title" }, "This source"),
        el("div", { class: "node is-self", "aria-current": "true" }, kindIcon(ctx.source.kind), el("span", { class: "node-title" }, ctx.source.title))),
      column("Children", children, related, "None"),
      column("Descendants", descendants, related, "None")),
    !nothing && el("section", { class: "family-siblings" },
      el("h4", { class: "family-title" }, "Siblings", el("span", { class: "family-note" }, "share a parent"), el("span", { class: "family-count" }, graph.siblings.length || "")),
      graph.siblings.length
        ? nodeList(graph.siblings, related)
        : el("p", { class: "quiet" }, "None")),
    others.length > 0 && el("section", { class: "link-groups" },
      el("h3", { class: "tab-title" }, "Other links"),
      others.map(([title, items]) => el("section", { class: "link-group" },
        el("h4", { class: "group-title" }, title, el("span", { class: "group-count" }, plural(items.length, "link"))),
        el("ul", { class: "link-list" }, items.map(({ card, note }) => el("li", { class: "link-item" }, nodeButton(card), el("p", { class: "link-note" }, note))))))));
}

export async function openRelationshipDialog({ source, trigger }) {
  const result = await openDialog({
    title: "Add relationship", trigger, size: "md", submitLabel: "Add relationship",
    description: "Read it as a sentence: this source … the other source. Links explain structure; they never approve a source.",
    build: () => {
      const relation = selectField({
        name: "relation", label: "This source…", options: SENTENCES.map(({ value, label }) => ({ value, label })),
        value: source.kind === "sources_of_sources" ? "contains" : "cites", hint: "",
      });
      const hint = el("p", { class: "callout", "aria-live": "polite" });
      const sync = () => { hint.textContent = SENTENCES.find((item) => item.value === relation.control.value)?.hint || ""; };
      relation.control.addEventListener("change", sync);
      sync();
      const other = sourcePicker({ name: "other", label: "Other source", required: true, exclude: [source.id] });
      const note = areaField({ name: "note", label: "Why are they related?", required: true, rows: 3, maxlength: 4000, hint: "Kept with the link so the reason can be audited." });
      return [relation.node, hint, other.node, note.node];
    },
    onSubmit: async (data) => {
      const choice = SENTENCES.find((item) => item.value === data.get("relation"));
      const other = String(data.get("other"));
      if (!other) throw new Error("Choose the other source from the results.");
      const note = String(data.get("note")).trim();
      const body = choice.reverse
        ? { source_id: other, target_id: source.id, kind: choice.kind, note }
        : { source_id: source.id, target_id: other, kind: choice.kind, note };
      return api("/relationships", { body });
    },
  });
  if (!result || result === true) return;
  emit("data-changed", { sourceId: source.id });
  toast({ message: "Relationship recorded." });
}
