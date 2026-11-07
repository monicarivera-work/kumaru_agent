// The network actions on a source (inspect, collect, check for editorial updates), with the
// requirements that gate them. The server enforces the same rules; this explains them.
import { api } from "./api.js";
import { toast, toastError } from "./dialogs.js";
import { ACCESS, plural } from "./dom.js";
import { emit } from "./state.js";

export function collectBlockers(source, summary) {
  const missing = [];
  if (source.kind === "sources_of_sources") missing.push("an individual work (containers are catalogues)");
  if (source.origin !== "human_verified") missing.push("verified human authorship");
  if (source.permission !== "text_permitted") missing.push("text-collection permission");
  if (!source.topic_reviewed) missing.push("reviewed topic coverage");
  if (summary.pending_notices) missing.push("no pending editorial notice");
  if (summary.withdrawn) missing.push("no retraction");
  if (!["unknown", "open"].includes(source.access)) missing.push("open or unknown access");
  return missing;
}

export const checkBlockers = (source) => (source.doi && source.kind !== "sources_of_sources" ? [] : ["a DOI on an individual work"]);

const ACTIONS = {
  inspect: {
    path: "inspect", busy: "Inspecting…", failure: "Inspection failed",
    done: (result) => ({ message: `Inspection finished. Access is now “${ACCESS[result.access]?.label || result.access}”.${result.access_note ? ` ${result.access_note}` : ""}` }),
  },
  collect: {
    path: "collect", busy: "Collecting…", failure: "Collection failed",
    done: (result) => ({ message: `Collected ${plural(result.word_count, "word")} of permitted text.` }),
  },
  check: {
    path: "check-updates", busy: "Checking…", failure: "Editorial check failed",
    done: (result) => ({
      message: `${result.notice_ids.length ? `${plural(result.notice_ids.length, "notice")} reported for review. ` : "No notices reported. "}${result.warning}`,
    }),
  },
};

export async function runSourceAction(kind, source, trigger) {
  const action = ACTIONS[kind];
  const label = trigger?.querySelector("span")?.textContent;
  if (trigger) { trigger.disabled = true; if (trigger.querySelector("span")) trigger.querySelector("span").textContent = action.busy; }
  try {
    const result = await api(`/sources/${source.id}/${action.path}`, { method: "POST" });
    toast({ message: action.done(result).message });
    emit("data-changed", { sourceId: source.id });
  } catch (error) {
    const outage = error.status === 503 ? " The source’s recorded access category was not changed." : "";
    toastError({ message: `${error.message}${outage}`, hint: error.hint }, action.failure);
  } finally {
    if (trigger?.isConnected) {
      trigger.disabled = false;
      if (label && trigger.querySelector("span")) trigger.querySelector("span").textContent = label;
    }
  }
}
