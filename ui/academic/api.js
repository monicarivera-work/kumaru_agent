// Fetch wrapper for the academic API. The key lives in this module's memory only: it is never
// written to storage, cookies or the address bar, so reloading the page forgets it.
let apiKey = "";

export const setApiKey = (value) => {
  const previous = apiKey;
  apiKey = value || "";
  return previous;
};
export const hasApiKey = () => apiKey !== "";

export class ApiError extends Error {
  constructor(message, { status = 0, type = "", hint = "", unreachable = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = type;
    this.hint = hint;
    this.unreachable = unreachable;
  }
}

function detailMessage(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      const where = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body" && part !== "query").join(".") : "";
      return where ? `${where}: ${item.msg}` : String(item.msg || item);
    }).join("; ");
  }
  return "";
}

export function query(values) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
  }
  const text = params.toString();
  return text ? `?${text}` : "";
}

export async function api(path, { method, body, params } = {}) {
  const verb = method || (body === undefined ? "GET" : "POST");
  const headers = {};
  if (apiKey) headers.Authorization = ["Bearer", apiKey].join(" ");
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(`/v1/academic${path}${params ? query(params) : ""}`, {
      method: verb, credentials: "same-origin", headers, ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    });
  } catch {
    throw new ApiError("Kumaru server unreachable. Check that it is running, then retry.", { unreachable: true });
  }
  let payload = null;
  if ((response.headers.get("content-type") || "").includes("json")) {
    try { payload = await response.json(); } catch { payload = null; }
  }
  if (response.status === 401) {
    throw new ApiError("API key missing or incorrect. Enter the configured server key and reconnect.", { status: 401 });
  }
  if (response.status === 404 && !payload?.error) {
    throw new ApiError("Academic API unavailable. Enable academic.enabled and restart Kumaru.", { status: 404 });
  }
  if (!response.ok) {
    const message = payload?.error?.message || detailMessage(payload?.detail) || `HTTP ${response.status} from server`;
    throw new ApiError(message, { status: response.status, type: payload?.error?.type || "", hint: payload?.error?.hint || "" });
  }
  if (payload === null) throw new ApiError(`HTTP ${response.status} from server`, { status: response.status });
  return payload;
}
