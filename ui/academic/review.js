// The Review source dialog: authorship, copying permission, access and topic coverage in one
// audited save. Only fields that actually changed are sent, so the history stays truthful.
import { api } from "./api.js";
import { openDialog, toast } from "./dialogs.js";
import { ACCESS, ACCESS_ORDER, el } from "./dom.js";
import { areaField, checkField, formSection, selectField, textField } from "./forms.js";
import { emit } from "./state.js";
import { topicCheckboxTree } from "./topics.js";

const sameSet = (a, b) => a.length === b.length && a.every((item) => b.includes(item));

function approval({ source, key, label, options, hint, affirmative, evidenceLabel, noteLabel }) {
  const decision = selectField({ name: key, label, value: source[key], options, hint });
  const url = textField({ name: `${key}_url`, label: evidenceLabel, type: "url", placeholder: "https://" });
  const note = areaField({ name: `${key}_note`, label: noteLabel, rows: 2, maxlength: 4000 });
  const evidence = el("div", { class: "conditional" }, url.node, note.node);
  const warning = el("p", { class: "callout is-warn" }, "Withdrawing this approval also deletes any collected text.");
  const sync = () => {
    const approving = decision.control.value === affirmative && source[key] !== affirmative;
    evidence.hidden = !approving;
    url.control.disabled = note.control.disabled = !approving;
    url.control.required = note.control.required = approving;
    warning.hidden = !(source[key] === affirmative && decision.control.value !== affirmative);
  };
  decision.control.addEventListener("change", sync);
  sync();
  return { nodes: [decision.node, evidence, warning], decision, url, note, approving: () => decision.control.value === affirmative && source[key] !== affirmative };
}

export async function openReviewDialog({ source, trigger, section = "" }) {
  const result = await openDialog({
    title: "Review source", trigger, size: "lg", submitLabel: "Save review",
    description: "Each change is an explicit human decision recorded in the audit history. Nothing here is approved automatically.",
    build: () => {
      const origin = approval({
        source, key: "origin", label: "Status", affirmative: "human_verified",
        options: [{ value: "unknown", label: "Not verified" }, { value: "human_verified", label: "Human verified" }, { value: "ai", label: "AI disclosed" }],
        hint: "An AI detector cannot certify human authorship. Cite what you actually checked.",
        evidenceLabel: "Authorship evidence URL", noteLabel: "Why this establishes human authorship",
      });
      const permission = approval({
        source, key: "permission", label: "Status", affirmative: "text_permitted",
        options: [{ value: "unknown", label: "Not reviewed" }, { value: "metadata_only", label: "Metadata only" }, { value: "text_permitted", label: "Text collection permitted" }],
        hint: "Free access or a subscription alone does not grant reuse rights.",
        evidenceLabel: "Licence or permission URL", noteLabel: "Permission and collection scope",
      });
      const access = selectField({
        name: "access", label: "Access category", value: source.access,
        options: ["unknown", ...ACCESS_ORDER.filter((value) => value !== "unknown")].map((value) => ({ value, label: ACCESS[value].label })),
        hint: "How the source can be read. Inspecting the page can refine this later.",
      });
      const accessNote = areaField({
        name: "access_note", label: "Access note", rows: 2, maxlength: 2000, value: source.access_note || "",
        hint: "Why the source is walled off, or how you reached it. Kept in the review history.",
      });
      const tree = topicCheckboxTree({ selected: source.topic_ids, legend: "Subjects this source covers" });
      const reviewed = checkField({
        name: "topic_reviewed", label: "I reviewed topic coverage against the material itself", checked: source.topic_reviewed,
        hint: "Tags are discovery candidates until you confirm them. Scores need confirmed coverage.",
      });
      const changed = el("p", { class: "callout is-warn", hidden: true }, "You changed the subjects, so coverage is unconfirmed. Tick the box again once you have checked the new tags.");
      tree.node.addEventListener("change", () => {
        const differs = !sameSet(tree.values(), source.topic_ids);
        reviewed.control.checked = differs ? false : source.topic_reviewed;
        changed.hidden = !differs;
      });
      const sections = [
        ["origin", "Human authorship", origin.nodes],
        ["permission", "Copying permission", permission.nodes],
        ["access", "Access", [access.node, accessNote.node]],
        ["topics", "Topics", [tree.node, changed, reviewed.node]],
      ];
      return sections.map(([key, title, children]) => {
        const node = formSection({ title, children });
        node.dataset.section = key;
        return node;
      });
    },
    onOpen: ({ form }) => {
      const target = form.querySelector(`[data-section="${section}"] select, [data-section="${section}"] input`);
      target?.focus();
    },
    onSubmit: async (data) => {
      const body = {};
      for (const key of ["origin", "permission", "access"]) {
        if (String(data.get(key)) !== source[key]) body[key] = String(data.get(key));
      }
      const note = String(data.get("access_note"));
      if (note !== (source.access_note || "")) body.access_note = note;
      const topics = data.getAll("topic_ids").map(String);
      const reviewed = data.has("topic_reviewed");
      if (!topics.length) throw new Error("Choose at least one subject for this source.");
      if (!sameSet(topics, source.topic_ids)) {
        body.topic_ids = topics;
        // Sent explicitly: the server treats a bare topic list as confirmed coverage.
        body.topic_reviewed = reviewed;
      } else if (reviewed !== source.topic_reviewed) {
        body.topic_reviewed = reviewed;
      }
      const evidence = [];
      for (const [key, kind] of [["origin", "origin"], ["permission", "permission"]]) {
        if (body[key] && ((key === "origin" && body[key] === "human_verified") || (key === "permission" && body[key] === "text_permitted"))) {
          evidence.push({ kind, value: 1, url: String(data.get(`${key}_url`)).trim(), note: String(data.get(`${key}_note`)).trim(), reviewed: true });
        }
      }
      if (evidence.length) body.evidence = evidence;
      if (Object.keys(body).length === 0) throw new Error("There is nothing to save yet. Change a field first.");
      return api(`/sources/${source.id}/review`, { body });
    },
  });
  if (!result || result === true) return;
  emit("data-changed", { sourceId: source.id });
  toast({ message: "Review saved to the local catalogue." });
}
