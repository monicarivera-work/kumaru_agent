// The two list views beside the Library: Sources of Sources grouped by access, and the review
// queue of everything that waits for a human decision.
import { api } from "./api.js";
import {
  ACCESS, ACCESS_ORDER, accessBadge, badge, button, clear, el, emptyState, fmtEventDate, fmtInt, kindIcon, NOTICE_KINDS, originBadge,
  plural, skeletonRows,
} from "./dom.js";
import { emit, gate, navigate, on, sourceHash, state, viewHash } from "./state.js";
import { topicName } from "./topics.js";

const PAGE = 30;
let hosts = {};
const containerRequests = gate();
const queueRequests = {};

/* ─── Sources of Sources ─── */

function browseContents(item) {
  Object.assign(state.filters, { access: "", kind: "", status: "", q: "", containerId: item.id, containerTitle: item.title, containerChildren: item.child_count });
  const search = document.getElementById("library-search");
  if (search) search.value = "";
  navigate(viewHash("library"));
  emit("filters-changed");
}

function containerRow(item) {
  return el("li", { class: "container-row" },
    el("div", { class: "container-main" },
      el("span", { class: "row-kind" }, kindIcon(item.kind)),
      el("div", { class: "container-text" },
        el("strong", { class: "container-title" }, item.title),
        el("span", { class: "row-meta" }, [item.publisher, plural(item.child_count, "work")].filter(Boolean).join(" · ")))),
    el("div", { class: "container-actions" },
      originBadge(item.origin),
      button("Browse contents", { iconName: "search", onclick: () => browseContents(item) }),
      button("Open", { onclick: () => navigate(sourceHash(item.id)) })));
}

function containerSection(access, token) {
  const info = ACCESS[access];
  const list = el("ul", { class: "container-list" });
  const more = el("div", { class: "section-more" });
  const count = el("span", { class: "section-count" }, "…");
  const body = el("div", { class: "section-body" }, skeletonRows(2));
  const node = el("section", { class: "view-section", "aria-labelledby": `containers-${access}` },
    el("header", { class: "view-section-head" },
      el("h3", { class: "view-section-title", id: `containers-${access}` }, accessBadge(access, { full: true }), count),
      el("p", { class: "view-section-meaning" }, info.meaning)),
    body);
  let loaded = 0;
  let total = 0;
  const fetchPage = async () => {
    try {
      const result = await api("/library", { params: { kind: "containers", access, sort: "title", limit: PAGE, offset: loaded } });
      if (!containerRequests.current(token)) return;
      total = result.total;
      loaded += result.items.length;
      count.textContent = fmtInt(total);
      list.append(...result.items.map(containerRow));
      more.replaceChildren(...(loaded < total ? [button(`Show ${Math.min(PAGE, total - loaded)} more`, { onclick: fetchPage })] : []));
      if (!total) body.replaceChildren(el("p", { class: "quiet" }, "None catalogued."));
      else body.replaceChildren(list, more);
    } catch (error) {
      if (!containerRequests.current(token)) return;
      count.textContent = "?";
      body.replaceChildren(emptyState({ tone: "error", title: "Could not load this group", body: error.message, actions: [button("Try again", { onclick: fetchPage })] }));
    }
  };
  fetchPage();
  return node;
}

export function showContainers() {
  const token = containerRequests.next();
  clear(hosts.containers, ACCESS_ORDER.map((access) => containerSection(access, token)));
}

/* ─── Review queue ─── */

function queueRow({ title, meta, badges, onclick, focusId }) {
  return el("li", {}, el("button", { type: "button", class: "queue-row", onclick, dataset: { focusId } },
    el("span", { class: "queue-main" }, el("span", { class: "queue-title" }, title), meta && el("span", { class: "row-meta" }, meta)),
    el("span", { class: "row-badges" }, badges)));
}

function queueSection({ key, title, countOf, fetchPage, row, empty }) {
  const requests = (queueRequests[key] ||= gate());
  let offset = 0;
  const body = el("div", { class: "section-body" });
  const count = el("span", { class: "section-count" }, "…");
  const node = el("section", { class: "view-section", "aria-labelledby": `queue-${key}` },
    el("header", { class: "view-section-head" }, el("h3", { class: "view-section-title", id: `queue-${key}` }, title, count)), body);
  const load = async () => {
    const token = requests.next();
    try {
      const { items, total } = await fetchPage(offset);
      if (!requests.current(token)) return;
      const known = Math.max(total ?? countOf(), offset + items.length);
      count.textContent = fmtInt(known);
      if (!items.length) {
        if (offset > 0) { offset = 0; load(); return; }
        body.replaceChildren(el("p", { class: "quiet" }, empty));
        return;
      }
      body.replaceChildren(
        el("ul", { class: "queue-list" }, items.map(row)),
        el("div", { class: "pager" },
          el("span", { class: "pager-range" }, `${fmtInt(offset + 1)}–${fmtInt(offset + items.length)} of ${fmtInt(known)}`),
          el("div", { class: "pager-buttons" },
            button("Previous", { disabled: offset === 0, onclick: () => { offset = Math.max(0, offset - PAGE); load(); } }),
            button("Next", { disabled: offset + PAGE >= known, onclick: () => { offset += PAGE; load(); } }))));
    } catch (error) {
      if (!requests.current(token)) return;
      count.textContent = "?";
      body.replaceChildren(emptyState({ tone: "error", title: "Could not load this list", body: error.message, actions: [button("Try again", { onclick: load })] }));
    }
  };
  body.append(skeletonRows(2));
  load();
  return node;
}

const snippet = (text) => (text.length > 140 ? `${text.slice(0, 137)}…` : text);

export function showQueue() {
  const counts = () => state.queue || { needs_review: 0, pending_notices: 0, unreviewed_claims: 0 };
  const jump = (key, label, count) => el("button", {
    type: "button", class: "btn btn-secondary",
    onclick: () => {
      const heading = document.getElementById(`queue-${key}`);
      heading.setAttribute("tabindex", "-1");
      heading.scrollIntoView({ block: "start" });
      heading.focus({ preventScroll: true });
    },
  }, el("span", {}, label), el("span", { class: "jump-count" }, fmtInt(count)));
  clear(hosts.queue,
    el("nav", { class: "queue-jump", "aria-label": "Queue sections" },
      jump("sources", "Sources awaiting review", counts().needs_review),
      jump("notices", "Pending notices", counts().pending_notices),
      jump("claims", "Unreviewed claims", counts().unreviewed_claims)),
    queueSection({
      key: "sources", title: "Sources awaiting human review", empty: "No sources are waiting. A quiet queue is not proof that every source has been checked well.",
      countOf: () => 0,
      fetchPage: async (offset) => {
        const result = await api("/library", { params: { status: "needs_review", sort: "updated", limit: PAGE, offset } });
        return { items: result.items, total: result.total };
      },
      row: (item) => queueRow({
        title: item.title, focusId: `queue-source-${item.id}`,
        meta: [item.origin === "unknown" && "Authorship not verified", !item.topic_reviewed && "Topic coverage not reviewed"].filter(Boolean).join(" · "),
        badges: [accessBadge(item.access)], onclick: () => navigate(sourceHash(item.id)),
      }),
    }),
    queueSection({
      key: "notices", title: "Pending editorial notices", empty: "No notices are pending. Crossref checks only cover works with a DOI, and a quiet queue is not proof that every source is current.",
      countOf: () => counts().pending_notices,
      fetchPage: async (offset) => ({ items: await api("/notices", { params: { state: "reported", limit: PAGE, offset } }) }),
      row: (item) => queueRow({
        title: item.notice.title, focusId: `queue-notice-${item.notice.id}`,
        meta: [item.source?.title, item.notice.event_date && fmtEventDate(item.notice.event_date)].filter(Boolean).join(" · "),
        badges: [badge({ tone: "warn", iconName: "alert", label: NOTICE_KINDS[item.notice.kind] || item.notice.kind })],
        onclick: () => navigate(sourceHash(item.notice.source_id, "notices")),
      }),
    }),
    queueSection({
      key: "claims", title: "Unreviewed claims", empty: "Every recorded claim has had its attribution reviewed. Claims are separate from their evidence, which is judged on its own.",
      countOf: () => counts().unreviewed_claims,
      fetchPage: async (offset) => ({ items: await api("/claims", { params: { reviewed: "false", limit: PAGE, offset } }) }),
      row: (item) => queueRow({
        title: snippet(item.claim.statement), focusId: `queue-claim-${item.claim.id}`,
        meta: [item.source?.title, item.claim.locator].filter(Boolean).join(" · "),
        badges: [el("span", { class: "tag" }, topicName(item.claim.topic_id))],
        onclick: () => navigate(sourceHash(item.claim.source_id, "claims", item.claim.id)),
      }),
    }));
}

export function initViews({ containers, queue }) {
  hosts = { containers, queue };
  const visible = () => document.body.dataset.view;
  on("data-changed", () => {
    if (visible() === "containers") showContainers();
    if (visible() === "queue") showQueue();
  });
}
