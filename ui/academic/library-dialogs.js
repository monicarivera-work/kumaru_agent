// Discover and Add-source dialogs. Neither approves anything: they create unreviewed candidates.
import { api } from "./api.js";
import { openDialog, toast, toastError } from "./dialogs.js";
import { ACCESS, ACCESS_ORDER, el, KINDS, plural } from "./dom.js";
import { radioGroup, segmented, selectField, textField } from "./forms.js";
import { sourcePicker } from "./picker.js";
import { emit, navigate, sourceHash, state } from "./state.js";
import { breadcrumb, breadcrumbText, topicCheckboxTree } from "./topics.js";

export async function openDiscoverDialog(trigger) {
  const topicId = state.topicId;
  const result = await openDialog({
    title: "Discover sources", trigger, size: "md", submitLabel: "Search",
    description: "Contacts an external provider; results are unreviewed candidates.",
    build: () => [
      el("p", { class: "dialog-subject" }, el("span", { class: "dialog-subject-label" }, "Subject"), breadcrumb(topicId)),
      textField({
        name: "query", label: "Narrow your search (optional)", maxlength: 500, placeholder: "For example: fundamental theorem",
        hint: "Added to the subject path, which is sent to the provider with it. Do not put secrets here.",
      }).node,
      radioGroup({
        name: "provider", legend: "Provider", value: "crossref",
        options: [
          { value: "crossref", label: "Crossref", hint: "Scholarly metadata: titles, authors, DOIs, citation counts." },
          { value: "brave", label: "Brave web search", disabled: !state.braveConfigured,
            hint: state.braveConfigured ? "Broader web results." : "Unavailable: academic.brave_api_key is not set in the Kumaru configuration." },
        ],
      }).node,
    ],
    onSubmit: (data) => api("/search", {
      body: { topic_id: topicId, query: String(data.get("query")).trim(), provider: String(data.get("provider")), limit: 20 },
    }),
  });
  if (!result || result === true) return;
  const skipped = result.skipped || 0;
  toast({
    title: "Discovery finished",
    message: `${plural(result.source_ids.length, "candidate")} added or refreshed for ${breadcrumbText(topicId)}`
      + `${skipped ? `; ${plural(skipped, "incomplete provider record")} skipped` : ""}. `
      + "Authorship, accuracy and copying rights still need your review.",
  });
  emit("data-changed", { scope: "sources" });
}

export async function openAddSourceDialog(trigger) {
  const result = await openDialog({
    title: "Add a source", trigger, size: "lg", submitLabel: "Catalogue source",
    description: "Adds metadata only. Authorship, rights and topic coverage stay unreviewed until you review them.",
    build: () => {
      const container = sourcePicker({
        name: "container_id", label: "Part of a container (optional)", kind: "containers",
        hint: "A journal, library or repository that lists this source.",
      });
      const note = textField({ name: "container_note", label: "Why is it part of that container?", maxlength: 4000, hint: "Required when a container is chosen." });
      note.control.disabled = true;
      note.node.hidden = true;
      container.node.addEventListener("change", () => {
        const chosen = Boolean(container.value());
        note.node.hidden = !chosen;
        note.control.disabled = !chosen;
        note.control.required = chosen;
      });
      return [
        textField({ name: "url", label: "Source URL", type: "url", required: true, placeholder: "https://" }).node,
        textField({ name: "title", label: "Title", required: true, maxlength: 2000 }).node,
        segmented({
          name: "kind", legend: "Kind", value: "website",
          options: Object.entries(KINDS).map(([value, info]) => ({ value, label: info.label })),
        }).node,
        textField({ name: "doi", label: "DOI (optional)", hint: "A DOI enables Crossref editorial-update checks.", placeholder: "10.1234/example" }).node,
        selectField({
          name: "access", label: "Access category", value: "unknown",
          options: ["unknown", ...ACCESS_ORDER.filter((value) => value !== "unknown")].map((value) => ({ value, label: ACCESS[value].label })),
          hint: "What you know about how this source can be read. Inspecting the page later can refine it.",
        }).node,
        topicCheckboxTree({ selected: [state.topicId], legend: "Subjects", hint: "Tags are candidates until you review topic coverage." }).node,
        container.node, note.node,
      ];
    },
    onSubmit: async (data) => {
      const topics = data.getAll("topic_ids");
      if (!topics.length) throw new Error("Choose at least one subject for this source.");
      const doi = String(data.get("doi")).trim();
      const source = await api("/sources", { body: {
        url: String(data.get("url")).trim(), title: String(data.get("title")).trim(), kind: String(data.get("kind")),
        doi: doi || null, access: String(data.get("access")), topic_ids: topics,
      } });
      const containerId = String(data.get("container_id") || "");
      if (!containerId) return { source };
      try {
        await api("/relationships", { body: {
          source_id: containerId, target_id: source.id, kind: "contains", note: String(data.get("container_note")).trim(),
        } });
        return { source };
      } catch (linkError) {
        return { source, linkError };
      }
    },
  });
  if (!result || result === true) return;
  emit("data-changed", { scope: "sources" });
  navigate(sourceHash(result.source.id, "overview"));
  if (result.linkError) {
    toastError(result.linkError, "Source catalogued, but it was not added to the container");
  } else {
    toast({ message: "Source catalogued as an unreviewed candidate. Review its authorship, rights and topic coverage before relying on it." });
  }
}
