# `ui/academic/picker.js`

A source picker: an ARIA combobox (input plus listbox) over `GET /library?q=&limit=8`,
debounced and generation-guarded, with Up/Down/Enter/Esc. It also accepts a pasted
24-hex ID. The choice is submitted through a hidden input and shown as a chip with
Change, so nobody types an ID. Required and half-typed states use custom validity, and
Esc closes the results without closing the dialog that holds the picker.
