// Small client for the EconoCausal REST API (backend/src/api.py).

export const API_URL = (import.meta.env?.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

const START_HINT =
  "Start it from the project root with: python -m uvicorn api:app --app-dir backend\\src --port 8000";

function describeError(status, body) {
  if (status === 422 && body && Array.isArray(body.detail)) {
    const first = body.detail
      .slice(0, 3)
      .map((d) => d.loc.slice(-1) + ": " + d.msg)
      .join("; ");
    return "The API rejected the data (" + first + ").";
  }
  if (body && typeof body.detail === "string") return body.detail;
  if (status === 503) {
    return "The API is running but has no model. Run python backend\\src\\train_final_model.py, then restart it.";
  }
  return "The API returned an error (" + status + ").";
}

async function request(path, options, baseUrl) {
  let res;
  try {
    res = await fetch(baseUrl + path, options);
  } catch {
    throw new Error("Could not reach the API at " + baseUrl + ". " + START_HINT);
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    // the response was not JSON
  }
  if (!res.ok) throw new Error(describeError(res.status, body));
  return body;
}

export function getHealth(baseUrl = API_URL) {
  return request("/health", { method: "GET" }, baseUrl);
}

export function checkDrift(customers, baseUrl = API_URL) {
  return request(
    "/drift",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customers }),
    },
    baseUrl
  );
}