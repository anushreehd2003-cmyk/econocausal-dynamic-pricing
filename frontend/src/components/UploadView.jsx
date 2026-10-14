import { useState } from "react";
import Papa from "papaparse";

const REQUIRED_COLUMNS = [
  "customer_id", "age", "income", "loyalty_member", "past_purchases",
  "days_since_last_purchase", "avg_basket_value", "discount",
  "purchased", "revenue",
];

export default function UploadView({ dataset, setDataset }) {
  const [error, setError] = useState("");

  function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    setError("");
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (result) => {
        const columns = result.meta.fields || [];
        const missing = REQUIRED_COLUMNS.filter((c) => !columns.includes(c));
        if (missing.length > 0) {
          setError("Missing columns: " + missing.join(", "));
          setDataset(null);
          return;
        }
        setDataset({ fileName: file.name, rows: result.data, columns });
      },
      error: (err) => setError(err.message),
    });
  }

  let summary = null;
  if (dataset) {
    const rows = dataset.rows;
    const count = (d) => rows.filter((r) => r.discount === d).length;
    const buyers = rows.filter((r) => r.purchased === 1).length;
    summary = {
      total: rows.length,
      none: count(0),
      d10: count(10),
      d20: count(20),
      rate: ((buyers / rows.length) * 100).toFixed(1),
    };
  }

  return (
    <div className="card">
      <h2>Upload historical campaign data</h2>
      <p>Choose a CSV with the columns listed in docs/data_dictionary.md.</p>
      <input type="file" accept=".csv" onChange={handleFile} />
      {error && <p className="error">{error}</p>}

      {dataset && summary && (
        <>
          <p className="ok">Loaded {dataset.fileName}</p>
          <div className="stats">
            <div className="stat"><strong>{summary.total}</strong>Customers</div>
            <div className="stat"><strong>{summary.none}</strong>No discount</div>
            <div className="stat"><strong>{summary.d10}</strong>$10 discount</div>
            <div className="stat"><strong>{summary.d20}</strong>$20 discount</div>
            <div className="stat"><strong>{summary.rate}%</strong>Purchase rate</div>
          </div>
          <h3>Preview (first 5 rows)</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>{dataset.columns.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {dataset.rows.slice(0, 5).map((r, i) => (
                  <tr key={i}>
                    {dataset.columns.map((c) => <td key={c}>{String(r[c])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}