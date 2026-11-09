# `ui/academic/forms.js`

Accessible form builders: text, textarea, select, checkbox, radio group, segmented
control, slider-plus-number and form sections. Each has a visible label, wires hints with
`aria-describedby` and returns its control so callers can react to changes. The range
field submits only the number input; the slider is a convenience. Radio options can be
disabled with a visible reason (for example Brave when no key is configured).
