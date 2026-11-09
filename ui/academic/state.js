// Shared state, a tiny event bus, stale-response gates and the unsaved-changes guard.
export const state = {
  connection: "connecting", // connecting | ok | key | offline
  topics: [],
  topicId: "mathematics",
  expanded: new Set(),
  overview: null, // counts for the selected subject
  queue: null, // catalogue-wide counts for the review queue and the header badge
  formula: null,
  braveConfigured: false,
  openalexKeyConfigured: false, // without a key OpenAlex shares a small keyless daily budget
  unpaywallConfigured: false, // Unpaywall needs academic.contact_email
  monitor: null,
  route: { view: "library", sourceId: null, tab: "overview", claimId: null },
  filters: { access: "", kind: "", status: "", q: "", sort: "updated", containerId: "", containerTitle: "", containerChildren: 0 },
};

const bus = new EventTarget();
export const emit = (name, detail = {}) => bus.dispatchEvent(new CustomEvent(name, { detail }));
export function on(name, handler) {
  const listener = (event) => handler(event.detail);
  bus.addEventListener(name, listener);
  return () => bus.removeEventListener(name, listener);
}

// A response is current only while no newer request of the same kind has started.
export function gate() {
  let latest = 0;
  return { next: () => ++latest, current: (token) => token === latest };
}

let dirtyCheck = () => false;
export const setDirtyCheck = (check) => { dirtyCheck = check; };
export const isDirty = () => dirtyCheck();

export const viewHash = (view) => `#/${view}`;
export function sourceHash(id, tab = "overview", claimId = "") {
  const parts = ["source", id];
  if (tab && tab !== "overview") parts.push(tab);
  if (claimId && tab === "claims") parts.push(claimId);
  return `#/${parts.join("/")}`;
}

// Pushes a history entry, or replaces the current one when only the tab or claim changed.
export function navigate(hash, { replace = false } = {}) {
  if (location.hash === hash) return false;
  if (replace) {
    history.replaceState(null, "", hash);
    emit("route-replaced");
  } else {
    location.hash = hash;
  }
  return true;
}

// Updates the address for in-place changes (an expanded claim) without re-rendering anything.
export function replaceHash(hash) {
  if (location.hash !== hash) history.replaceState(null, "", hash);
}
