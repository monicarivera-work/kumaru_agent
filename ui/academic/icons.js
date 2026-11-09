// Inline 16px icons, 1.5px stroke, drawn in currentColor. Built with createElementNS only.
const NS = "http://www.w3.org/2000/svg";

// ["path", d] is a stroked path, ["fill", d] a solid shape; circle and rect carry their numbers.
const SHAPES = {
  article: [["path", "M3.5 1.75h6l3 3v9.5h-9z"], ["path", "M9.5 1.75v3h3"], ["path", "M5.5 8h5M5.5 10.5h5"]],
  book: [
    ["path", "M2.5 2.5h3.7A1.8 1.8 0 0 1 8 4.3v9.2a1.3 1.3 0 0 0-1.3-1.2H2.5z"],
    ["path", "M13.5 2.5H9.8A1.8 1.8 0 0 0 8 4.3v9.2a1.3 1.3 0 0 1 1.3-1.2h4.2z"],
  ],
  study: [
    ["path", "M6 2h4M6.75 2v4.5L3.2 12.8a1 1 0 0 0 .9 1.45h7.8a1 1 0 0 0 .9-1.45L9.25 6.5V2"],
    ["path", "M4.7 10.5h6.6"],
  ],
  globe: [
    ["circle", 8, 8, 6], ["path", "M2 8h12"],
    ["path", "M8 2c2 1.8 2.8 3.8 2.8 6S10 12.2 8 14c-2-1.8-2.8-3.8-2.8-6S6 3.8 8 2z"],
  ],
  library: [["path", "M2 5.5 8 2l6 3.5z"], ["path", "M3.5 7v5.5M6.5 7v5.5M9.5 7v5.5M12.5 7v5.5"], ["path", "M2 14h12"]],
  unlock: [["rect", 3, 7, 10, 7, 1.5], ["path", "M5.5 7V5a2.5 2.5 0 0 1 4.8-1"]],
  user: [["circle", 8, 5.5, 2.5], ["path", "M3 13.5c.5-2.6 2.6-4 5-4s4.5 1.4 5 4"]],
  lock: [["rect", 3, 7, 10, 7, 1.5], ["path", "M5.5 7V5a2.5 2.5 0 0 1 5 0v2"]],
  ban: [["circle", 8, 8, 6], ["path", "M3.8 3.8l8.4 8.4"]],
  question: [["circle", 8, 8, 6], ["path", "M6.2 6.3a1.9 1.9 0 1 1 2.8 1.7c-.7.4-1 .8-1 1.5"], ["path", "M8 11.4v.1"]],
  check: [["path", "M3 8.5l3.2 3.2L13 4.8"]],
  alert: [["path", "M8 2.2l6.2 11H1.8z"], ["path", "M8 6.5v3"], ["path", "M8 11.5v.1"]],
  info: [["circle", 8, 8, 6], ["path", "M8 7.2v4"], ["path", "M8 4.9v.1"]],
  external: [["path", "M6.5 3H3v10h10V9.5"], ["path", "M9.5 2.5h4v4"], ["path", "M13.5 2.5l-6 6"]],
  chevron: [["path", "M6 3.5 10.5 8 6 12.5"]],
  plus: [["path", "M8 3v10M3 8h10"]],
  copy: [
    ["rect", 5.5, 5.5, 8, 8, 1.5],
    ["path", "M10.5 5.5V4A1.5 1.5 0 0 0 9 2.5H4A1.5 1.5 0 0 0 2.5 4v5A1.5 1.5 0 0 0 4 10.5h1.5"],
  ],
  search: [["circle", 7, 7, 4.5], ["path", "M10.5 10.5 14 14"]],
  x: [["path", "M3.5 3.5l9 9M12.5 3.5l-9 9"]],
  back: [["path", "M13 8H3M7 4 3 8l4 4"]],
  edit: [
    ["path", "M2.5 13.5l.8-3.3 7.6-7.6a1.4 1.4 0 0 1 2 0l.5.5a1.4 1.4 0 0 1 0 2l-7.6 7.6z"],
    ["path", "M9.5 4l2.5 2.5"],
  ],
  menu: [["path", "M2.5 4.5h11M2.5 8h11M2.5 11.5h11"]],
  clock: [["circle", 8, 8, 6], ["path", "M8 4.5V8l2.5 1.5"]],
  link: [
    ["path", "M6.7 9.3a2.6 2.6 0 0 0 3.7 0l2-2a2.6 2.6 0 0 0-3.7-3.7l-.7.7"],
    ["path", "M9.3 6.7a2.6 2.6 0 0 0-3.7 0l-2 2a2.6 2.6 0 0 0 3.7 3.7l.7-.7"],
  ],
  filter: [["path", "M2 3.5h12L9.5 9v4l-3-1.5V9z"]],
  up: [["fill", "M8 3.5 13 11.5H3z"]],
  down: [["fill", "M8 12.5 3 4.5h10z"]],
  dot: [["fill", "M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"]],
};

export function icon(name, { size = 16, label = "" } = {}) {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.5");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("class", `icon icon-${name}`);
  svg.setAttribute("focusable", "false");
  if (label) {
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", label);
  } else {
    svg.setAttribute("aria-hidden", "true");
  }
  for (const [kind, ...values] of SHAPES[name] || []) {
    const shape = document.createElementNS(NS, kind === "fill" ? "path" : kind);
    if (kind === "path" || kind === "fill") shape.setAttribute("d", values[0]);
    if (kind === "fill") {
      shape.setAttribute("fill", "currentColor");
      shape.setAttribute("stroke", "none");
    } else if (kind === "circle") {
      shape.setAttribute("cx", String(values[0]));
      shape.setAttribute("cy", String(values[1]));
      shape.setAttribute("r", String(values[2]));
    } else if (kind === "rect") {
      ["x", "y", "width", "height", "rx"].forEach((name, index) => shape.setAttribute(name, String(values[index])));
    }
    svg.append(shape);
  }
  return svg;
}
