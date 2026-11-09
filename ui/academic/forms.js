// Accessible form controls: every control has a visible label, hints are wired with
// aria-describedby, and each builder returns the control so callers can react to it.
import { el, uid } from "./dom.js";

function describe(control, hintId) {
  if (!hintId) return;
  const existing = control.getAttribute("aria-describedby");
  control.setAttribute("aria-describedby", existing ? `${existing} ${hintId}` : hintId);
}

function wrap(label, control, { hint = "", required = false, className = "" } = {}) {
  const hintId = hint ? `${control.id}-hint` : "";
  describe(control, hintId);
  const node = el("div", { class: ["field", className] },
    el("label", { for: control.id, class: "field-label" }, label, required && el("span", { class: "required-mark", "aria-hidden": "true" }, " *")),
    control,
    hint && el("p", { class: "field-hint", id: hintId }, hint));
  return { node, control };
}

export function textField({ name, label, value = "", type = "text", required = false, hint = "", placeholder = "", maxlength, pattern, inputmode, readOnly = false, className = "" }) {
  const control = el("input", {
    id: uid("f"), name, type, value, required, placeholder, maxlength, pattern, inputmode, readOnly,
    autocomplete: "off", spellcheck: type === "text" ? "true" : "false", class: "input",
  });
  return wrap(label, control, { hint, required, className });
}

export function areaField({ name, label, value = "", rows = 3, required = false, hint = "", maxlength, placeholder = "", className = "" }) {
  const control = el("textarea", { id: uid("f"), name, rows, required, maxlength, placeholder, class: "input textarea" });
  control.value = value;
  return wrap(label, control, { hint, required, className });
}

export function selectField({ name, label, options, value = "", required = false, hint = "", className = "" }) {
  const control = el("select", { id: uid("f"), name, required, class: "input select" },
    options.map((option) => el("option", { value: option.value, disabled: option.disabled, selected: option.value === value }, option.label)));
  return wrap(label, control, { hint, required, className });
}

export function checkField({ name, label, checked = false, hint = "", value = "on", disabled = false }) {
  const control = el("input", { id: uid("f"), type: "checkbox", name, value, checked, disabled, class: "check-input" });
  const hintId = hint ? `${control.id}-hint` : "";
  describe(control, hintId);
  const node = el("div", { class: "check-field" },
    el("label", { class: "checkbox", for: control.id }, control, el("span", { class: "check-text" }, label)),
    hint && el("p", { class: "field-hint", id: hintId }, hint));
  return { node, control };
}

// A radio group inside a fieldset; options may be disabled with a visible reason.
export function radioGroup({ name, legend, options, value = "", hint = "", required = false, className = "", layout = "stack" }) {
  const hintId = hint ? uid("hint") : "";
  const inputs = [];
  const items = options.map((option) => {
    const input = el("input", { id: uid("r"), type: "radio", name, value: option.value, checked: option.value === value, disabled: option.disabled, required, class: "check-input" });
    inputs.push(input);
    return el("label", { class: ["choice", option.disabled && "is-disabled"], for: input.id },
      input, el("span", { class: "choice-text" }, el("span", { class: "choice-label" }, option.label),
        option.hint && el("span", { class: "choice-hint" }, option.hint)));
  });
  const node = el("fieldset", { class: ["field", "choice-group", `is-${layout}`, className], "aria-describedby": hintId || undefined },
    el("legend", { class: "field-label" }, legend),
    el("div", { class: "choice-list" }, items),
    hint && el("p", { class: "field-hint", id: hintId }, hint));
  return { node, inputs, value: () => inputs.find((input) => input.checked)?.value || "" };
}

// Radios drawn as one segmented control; still a native radio group for keyboards and readers.
export function segmented({ name, legend, options, value = "", hint = "" }) {
  const hintId = hint ? uid("hint") : "";
  const inputs = [];
  const segments = options.map((option) => {
    const input = el("input", { id: uid("s"), type: "radio", name, value: option.value, checked: option.value === value, class: "segment-input" });
    inputs.push(input);
    return el("label", { class: "segment", for: input.id }, input, el("span", {}, option.label));
  });
  const node = el("fieldset", { class: "field segmented", "aria-describedby": hintId || undefined },
    el("legend", { class: "field-label" }, legend), el("div", { class: "segment-row" }, segments),
    hint && el("p", { class: "field-hint", id: hintId }, hint));
  return { node, inputs, value: () => inputs.find((input) => input.checked)?.value || "" };
}

// A slider for quick adjustment plus a number input for precision; only the number is submitted.
export function rangeField({ name, label, value = 1, min = 0, max = 1, step = 0.01, hint = "", required = true, disabled = false }) {
  const number = el("input", { id: uid("f"), name, type: "number", min, max, step, value, required, disabled, inputmode: "decimal", class: "input number" });
  const slider = el("input", {
    type: "range", min, max, step, value, disabled, class: "range", "aria-label": `${label} (slider)`, tabindex: -1,
  });
  slider.addEventListener("input", () => { number.value = slider.value; number.dispatchEvent(new Event("input", { bubbles: true })); });
  number.addEventListener("input", () => { if (number.value !== "") slider.value = number.value; });
  const field = wrap(label, number, { hint, required });
  field.node.classList.add("range-field");
  field.node.querySelector(".field-label").after(el("div", { class: "range-row" }, slider, number));
  return { ...field, slider, setValue: (next) => { number.value = String(next); slider.value = String(next); } };
}

export function formSection({ title, hint = "", children }) {
  const hintId = hint ? uid("hint") : "";
  return el("fieldset", { class: "form-section", "aria-describedby": hintId || undefined },
    el("legend", { class: "form-section-title" }, title),
    el("div", { class: "form-section-body" },
      hint && el("p", { class: "field-hint", id: hintId }, hint), children));
}

export const fieldRow = (...children) => el("div", { class: "field-row" }, children);
