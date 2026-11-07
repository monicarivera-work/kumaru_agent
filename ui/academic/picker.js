// Source picker: an ARIA combobox over the catalogue search, so nobody types 24-hex IDs.
// A pasted ID also works. The chosen ID is submitted through a hidden input.
import { api } from "./api.js";
import { accessBadge, button, el, kindIcon, uid } from "./dom.js";
import { gate } from "./state.js";

const ID_PATTERN = /^[0-9a-f]{24}$/i;

export function sourcePicker({ name, label, required = false, hint = "", kind = "", exclude = [], initial = null, placeholder = "Search by title, author or DOI, or paste a source ID" }) {
  const inputId = uid("picker");
  const listId = uid("picker-list");
  const statusId = uid("picker-status");
  const hintId = hint ? uid("hint") : "";
  const requests = gate();
  const hidden = el("input", { type: "hidden", name, value: initial?.id || "" });
  const input = el("input", {
    id: inputId, type: "text", class: "input", role: "combobox", "aria-autocomplete": "list", "aria-expanded": "false",
    "aria-controls": listId, "aria-describedby": [hintId, statusId].filter(Boolean).join(" "), autocomplete: "off",
    spellcheck: "false", placeholder,
  });
  const list = el("ul", { id: listId, role: "listbox", class: "picker-list", hidden: true, "aria-label": `${label} results` });
  const status = el("p", { id: statusId, class: "picker-status", role: "status" });
  const chosen = el("div", { class: "picker-chosen", hidden: true });
  const search = el("div", { class: "picker-search" }, input, list);
  let selected = null;
  let options = [];
  let active = -1;
  let timer = 0;

  const syncValidity = () => {
    const typed = input.value.trim() !== "";
    input.setCustomValidity(!selected && (required || typed) ? "Choose a source from the results." : "");
  };

  function close() {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    active = -1;
  }

  function highlight(index) {
    active = index;
    [...list.children].forEach((item, position) => item.setAttribute("aria-selected", String(position === index)));
    if (index >= 0) {
      input.setAttribute("aria-activedescendant", list.children[index].id);
      list.children[index].scrollIntoView({ block: "nearest" });
    } else input.removeAttribute("aria-activedescendant");
  }

  function renderOptions() {
    list.replaceChildren(...options.map((card, index) => el("li", {
      id: `${listId}-${index}`, role: "option", class: "picker-option", "aria-selected": "false",
      onmousedown: (event) => { event.preventDefault(); choose(card); },
    }, kindIcon(card.kind), el("span", { class: "picker-option-text" },
      el("span", { class: "picker-option-title" }, card.title),
      el("span", { class: "picker-option-meta" }, [card.publisher, card.published_year].filter(Boolean).join(" · "))),
    accessBadge(card.access))));
    list.hidden = options.length === 0;
    input.setAttribute("aria-expanded", String(options.length > 0));
    highlight(options.length ? 0 : -1);
  }

  function renderChosen() {
    chosen.hidden = !selected;
    search.hidden = Boolean(selected);
    hidden.value = selected?.id || "";
    if (!selected) { chosen.replaceChildren(); return; }
    chosen.replaceChildren(
      kindIcon(selected.kind),
      el("span", { class: "picker-chosen-title" }, selected.title),
      accessBadge(selected.access),
      button("Change", { variant: "quiet", onclick: () => { select(null); input.focus(); } }));
  }

  function select(card) {
    selected = card;
    renderChosen();
    syncValidity();
    close();
    status.textContent = card ? `Selected ${card.title}.` : "";
    hidden.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function choose(card) {
    input.value = "";
    select(card);
  }

  async function lookup() {
    const text = input.value.trim();
    const token = requests.next();
    if (!text) { options = []; close(); status.textContent = ""; return; }
    status.textContent = "Searching…";
    try {
      if (ID_PATTERN.test(text)) {
        const detail = await api(`/sources/${text.toLowerCase()}`);
        if (!requests.current(token)) return;
        options = [{ ...detail.source }].filter((card) => !exclude.includes(card.id));
      } else {
        const result = await api("/library", { params: { q: text, kind, limit: 8, sort: "title" } });
        if (!requests.current(token)) return;
        options = result.items.filter((card) => !exclude.includes(card.id));
      }
      renderOptions();
      status.textContent = options.length ? `${options.length} ${options.length === 1 ? "result" : "results"}. Use the arrow keys to choose.` : "No matching sources.";
    } catch (error) {
      if (!requests.current(token)) return;
      options = [];
      renderOptions();
      status.textContent = /Unknown academic source/.test(error.message) ? "No source has this ID." : error.message;
    }
  }

  input.addEventListener("input", () => {
    syncValidity();
    clearTimeout(timer);
    timer = setTimeout(lookup, 250);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (!options.length) return;
      event.preventDefault();
      if (list.hidden) { list.hidden = false; input.setAttribute("aria-expanded", "true"); }
      highlight((active + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length);
    } else if (event.key === "Enter" && !list.hidden && active >= 0) {
      event.preventDefault();
      choose(options[active]);
    } else if (event.key === "Escape" && !list.hidden) {
      // Closing the results must not also close the dialog that contains the picker.
      event.preventDefault();
      event.stopPropagation();
      close();
    }
  });
  input.addEventListener("blur", () => setTimeout(close, 120));

  const node = el("div", { class: "field picker" },
    el("label", { for: inputId, class: "field-label" }, label, required && el("span", { class: "required-mark", "aria-hidden": "true" }, " *")),
    hidden, search, chosen, status,
    hint && el("p", { class: "field-hint", id: hintId }, hint));
  if (initial) selected = initial;
  renderChosen();
  syncValidity();
  return { node, input, value: () => selected?.id || "", card: () => selected, choose, focus: () => (selected ? chosen.querySelector("button") : input).focus() };
}
