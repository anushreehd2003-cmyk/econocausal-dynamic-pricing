import { useEffect, useState } from "react";
import { API_URL, getHealth, checkDrift } from "../utils/api";
import { MIN_ROWS, toDriftPayload } from "../utils/drift";

const LABELS = { ok: "OK", warning: "Warning", alert: "Alert" };

export default function DriftPanel({ dataset }) {
  const [health, setHealth] = useState({ state: "checking", info: null, message: "" });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((info) => {
        if (!cancelled) setHealth({ state: "up", info, message: "" });
      })
      .catch((e) => {
        if (!cancelled) setHealth({ state: "down", info: null, message: e.message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function runCheck() {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const payload = toDriftPayload(dataset.rows);
      if (payload.customers.length < MIN_ROWS) {
        throw new Error(
          "A drift check needs at least " + MIN_ROWS + " valid customers; this file has " +
            payload.customers.length + "."
        );
      }
      const report = await checkDrift(payload.customers);
      setResult({ report, payload });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const report = result ? result.report : null;
  const payload = result ? result.payload : null;

  return (
    <div className="card">
      <h2>Data health check</h2>
      <p>
        Compare this file with the customers the model was trained on. The
        check warns when customers or their purchase behaviour have changed.
      </p>

      <p className="status-line">
        {health.state === "checking" && "Checking the API..."}
        {health.state === "up" && (
          <span className="ok">
            API connected ({health.info.model_kind} model
            {health.info.drift_reference_available ? "" : ", no drift reference: retrain the model"})
          </span>
        )}
        {health.state === "down" && <span className="error">{health.message}</span>}
      </p>

      <button className="primary" onClick={runCheck} disabled={busy}>
        {busy ? "Checking..." : "Check data health"}
      </button>
      {error && <p className="error">{error}</p>}

      {report && (
        <div className="drift-result">
          <p>
            <span className={"badge " + report.status}>{LABELS[report.status]}</span>{" "}
            {report.recommendation}
          </p>
          {report.messages.length > 0 && (
            <ul>
              {report.messages.map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
          )}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Check</th>
                  <th>Variable</th>
                  <th className="num">PSI</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {report.items.map((item) => (
                  <tr key={item.kind + item.name}>
                    <td>{item.kind}</td>
                    <td>{item.name}</td>
                    <td className="num">{item.psi.toFixed(3)}</td>
                    <td>
                      <span className={"badge small " + item.severity}>{LABELS[item.severity]}</span>
                    </td>
                  </tr>
                ))}
                {report.behaviour && (
                  <tr>
                    <td>behaviour</td>
                    <td>purchase rate without a discount</td>
                    <td className="num">
                      {(report.behaviour.reference_rate * 100).toFixed(1)}% to{" "}
                      {(report.behaviour.current_rate * 100).toFixed(1)}%
                    </td>
                    <td>
                      <span className={"badge small " + report.behaviour.severity}>
                        {LABELS[report.behaviour.severity]}
                      </span>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="small">
            Checked {report.n_current.toLocaleString()} customers against {report.n_reference.toLocaleString()} training customers.
            {payload.skipped > 0 && " " + payload.skipped + " rows with missing or invalid values were left out."}
            {payload.truncated && " Only the first 10,000 valid rows were sent."}
            {report.skipped.length > 0 && " Not checked: " + report.skipped.join("; ") + "."}
          </p>
          <p className="small">
            PSI below 0.10 is stable, 0.10 to 0.25 is a moderate shift, and above
            0.25 is a large shift. The purchase-rate check uses customers who
            received no discount. Talking to {API_URL}.
          </p>
        </div>
      )}
    </div>
  );
}