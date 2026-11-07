// Native <dialog> helpers: form dialogs with inline errors and a dirty guard, a confirmation
// dialog, toasts, and focus restoration for the control that opened them.
import { append, button, el, iconButton, uid } from "./dom.js";
import { icon } from "./icons.js";
import { setDirtyCheck } from "./state.js";

const controllers = new Set();
setDirtyCheck(() => [...controllers].some((controller) => controller.isDirty()));

export function closeAllDialogs() {
  for (const controller of [...controllers]) controller.close(undefined, { restoreFocus: false });
}

// After a save the control that opened a dialog is often rebuilt; renderers call this once
// their DOM is in place so focus returns to the matching control instead of the page top.
let pendingFocus = null;
export function rememberFocus(focusId) {
  pendingFocus = focusId ? { focusId, until: Date.now() + 4000 } : null;
}
export function applyPendingFocus() {
  if (!pendingFocus) return;
  if (Date.now() > pendingFocus.until) { pendingFocus = null; return; }
  const node = document.querySelector(`[data-focus-id="${CSS.escape(pendingFocus.focusId)}"]`);
  if (!node) return;
  pendingFocus = null;
  if (!node.disabled) node.focus();
}

function snapshot(form) {
  return JSON.stringify([...new FormData(form)].map(([key, value]) => [key, typeof value === "string" ? value : value.name]));
}

export function confirmDialog({ title, body, confirmLabel = "Confirm", cancelLabel = "Cancel", danger = false }) {
  return new Promise((resolve) => {
    const titleId = uid("confirm-title");
    const bodyId = uid("confirm-body");
    const keep = el("button", { type: "submit", value: "cancel", class: "btn btn-secondary", autofocus: true }, cancelLabel);
    const accept = el("button", { type: "submit", value: "confirm", class: ["btn", danger ? "btn-danger" : "btn-primary"] }, confirmLabel);
    const dialog = el("dialog", { class: "ac-dialog is-sm", role: "alertdialog", "aria-labelledby": titleId, "aria-describedby": bodyId },
      el("form", { method: "dialog", class: "dialog-form" },
        el("div", { class: "dialog-head" }, el("h2", { id: titleId, class: "dialog-title" }, title)),
        el("div", { class: "dialog-body" }, el("p", { id: bodyId }, body)),
        el("div", { class: "dialog-foot" }, keep, accept)));
    dialog.addEventListener("close", () => {
      const confirmed = dialog.returnValue === "confirm";
      dialog.remove();
      resolve(confirmed);
    });
    document.body.append(dialog);
    dialog.showModal();
  });
}

export const confirmDiscard = () => confirmDialog({
  title: "Discard unsaved changes?",
  body: "What you entered in this dialog has not been saved.",
  confirmLabel: "Discard changes", cancelLabel: "Keep editing", danger: true,
});

/**
 * Opens a modal form. `build(controller)` returns the body; `onSubmit(formData, controller)` saves
 * and may throw: the message appears inline and every input stays as the user left it.
 * Resolves with onSubmit's result (true if it returns nothing), or undefined when cancelled.
 */
export function openDialog({ title, description = "", build, onSubmit, onOpen = null, submitLabel = "Save", cancelLabel = "Cancel", size = "md", trigger = null, hideSubmit = false }) {
  return new Promise((resolve) => {
    const titleId = uid("dialog-title");
    const focusId = trigger?.dataset?.focusId || "";
    const errorBox = el("div", { class: "dialog-error", role: "alert" });
    const fields = el("fieldset", { class: "dialog-fields" });
    const submit = el("button", { type: "submit", class: "btn btn-primary" }, el("span", {}, submitLabel));
    const form = el("form", { class: "dialog-form" });
    const dialog = el("dialog", { class: ["ac-dialog", `is-${size}`], "aria-labelledby": titleId });
    let busy = false;
    let closed = false;
    let confirming = false;
    let baseline = "";

    const controller = {
      dialog, form,
      isDirty: () => !closed && !busy && snapshot(form) !== baseline,
      rebaseline: () => { baseline = snapshot(form); },
      setError: (message) => { errorBox.textContent = message || ""; },
      close,
    };

    function close(result, { restoreFocus = true } = {}) {
      if (closed) return;
      closed = true;
      controllers.delete(controller);
      dialog.close();
      dialog.remove();
      if (restoreFocus) {
        if (trigger?.isConnected) trigger.focus();
        rememberFocus(focusId);
      }
      resolve(result);
    }

    async function requestClose() {
      if (busy || closed || confirming) return;
      if (controller.isDirty()) {
        confirming = true;
        const discard = await confirmDiscard();
        confirming = false;
        if (!discard) return;
      }
      close(undefined);
    }

    function setBusy(value) {
      busy = value;
      fields.disabled = value;
      submit.setAttribute("aria-disabled", String(value));
      submit.classList.toggle("is-busy", value);
      submit.firstElementChild.textContent = value ? "Saving…" : submitLabel;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (busy || !onSubmit) return;
      if (!form.reportValidity()) return;
      const data = new FormData(form);
      const active = document.activeElement;
      controller.setError("");
      setBusy(true);
      try {
        const result = await onSubmit(data, controller);
        close(result === undefined ? true : result);
      } catch (error) {
        setBusy(false);
        controller.setError(error?.message || String(error));
        errorBox.scrollIntoView({ block: "nearest" });
        if (active && active !== document.body && active.isConnected) active.focus();
      }
    });
    dialog.addEventListener("cancel", (event) => { event.preventDefault(); requestClose(); });
    dialog.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !event.defaultPrevented) { event.preventDefault(); requestClose(); }
    });

    fields.append(...[build(controller)].flat(Infinity).filter(Boolean));
    form.append(
      el("div", { class: "dialog-head" }, el("h2", { id: titleId, class: "dialog-title" }, title),
        iconButton("x", "Close dialog", { onclick: requestClose })),
      el("div", { class: "dialog-body" }, description && el("p", { class: "dialog-lede" }, description), fields, errorBox),
      el("div", { class: "dialog-foot" },
        button(cancelLabel, { variant: "secondary", onclick: requestClose }), !hideSubmit && submit));
    dialog.append(form);
    document.body.append(dialog);
    controllers.add(controller);
    dialog.showModal();
    controller.rebaseline();
    // Start in the first field of the form rather than on the close button.
    fields.querySelector("input:not([type='hidden']):not(:disabled), select:not(:disabled), textarea:not(:disabled)")?.focus();
    onOpen?.(controller);
  });
}

/* ─── Toasts ─── */

export function toast({ message, kind = "success", title = "", details = "" }) {
  const host = document.getElementById("toasts");
  if (!host) return;
  const failure = kind === "error";
  const node = el("div", { class: ["toast", `is-${kind}`], role: failure ? "alert" : "status" });
  const remove = () => node.remove();
  const content = () => append(node, [
    icon(failure ? "alert" : "check", { size: 18 }),
    el("div", { class: "toast-body" },
      title && el("strong", { class: "toast-title" }, title),
      el("p", { class: "toast-message" }, message),
      details && el("details", { class: "toast-details" }, el("summary", {}, "Show details"), el("pre", {}, details))),
    failure && button("Dismiss", { variant: "quiet", onclick: remove })]);
  host.append(node);
  // A live region announces content added after it exists, so fill it on the next frame.
  requestAnimationFrame(content);
  if (!failure) {
    let timer = setTimeout(remove, 5000);
    node.addEventListener("mouseenter", () => clearTimeout(timer));
    node.addEventListener("mouseleave", () => { timer = setTimeout(remove, 2500); });
  }
}

export const toastError = (error, title = "") => toast({ kind: "error", title, message: error?.message || String(error), details: error?.hint || "" });
