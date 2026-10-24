import { useEffect, useMemo, useState } from "react";
import PlotlyChart from "./PlotlyChart";
import { loadQini, loadCustomers } from "../utils/demoData";
import { filterCustomers, summarize, histogram } from "../utils/ite";
import { purchaseRateByDiscount } from "../utils/stats";

const COLORS = ["#2f6fed", "#12a594", "#d64545"];
const BINS = 40;

function points(value) {
  const pts = value * 100;
  return (pts >= 0 ? "+" : "") + pts.toFixed(1) + " pts";
}

export default function UpliftView({ dataset }) {
  const [qini, setQini] = useState(null);
  const [customers, setCustomers] = useState(null);
  const [loadMs, setLoadMs] = useState(null);
  const [error, setError] = useState("");
  const [level, setLevel] = useState(20);
  const [filters, setFilters] = useState({
    loyalty: "all",
    recencyMin: 0,
    recencyMax: 365,
    heldOutOnly: false,
  });

  useEffect(() => {
    let cancelled = false;
    const start = performance.now();
    Promise.all([loadQini(), loadCustomers()])
      .then(([q, c]) => {
        if (cancelled) return;
        setQini(q);
        setCustomers(c);
        setLoadMs(Math.round(performance.now() - start));
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const key = "ite_" + level;

  const range = useMemo(() => {
    if (!customers) return { lo: -0.3, hi: 0.3 };
    let lo = Infinity;
    let hi = -Infinity;
    for (const r of customers) {
      if (r[key] < lo) lo = r[key];
      if (r[key] > hi) hi = r[key];
    }
    return { lo, hi };
  }, [customers, key]);

  const processed = useMemo(() => {
    if (!customers) return null;
    const t0 = performance.now();
    const rows = filterCustomers(customers, filters);
    const stats = summarize(rows, key);
    const hist = histogram(rows.map((r) => r[key]), range.lo, range.hi, BINS);
    return { rows, stats, hist, ms: performance.now() - t0 };
  }, [customers, filters, key, range]);

  const qiniChart = useMemo(() => {
    if (!qini) return null;
    const data = Object.keys(qini.curves).map((name, i) => ({
      x: qini.x_percent,
      y: qini.curves[name],
      type: "scatter",
      mode: "lines",
      name,
      line: {
        color: COLORS[i % COLORS.length],
        width: name === qini.selected_model ? 4 : 2,
      },
    }));
    data.push({
      x: qini.x_percent, y: qini.random, type: "scatter", mode: "lines",
      name: "Random targeting", line: { dash: "dash", color: "#7b8794" },
    });
    data.push({
      x: qini.x_percent, y: qini.best_possible, type: "scatter", mode: "lines",
      name: "Best possible ranking", line: { dash: "dot", color: "#1f2933" },
    });
    return {
      data,
      layout: {
        title: "Qini curves, $" + qini.discount + " discount (held-out customers)",
        xaxis: { title: "% of customers targeted (highest score first)" },
        yaxis: { title: "Extra purchases per 1,000 customers" },
        legend: { orientation: "h", y: -0.35 },
      },
    };
  }, [qini]);

  const histChart = useMemo(() => {
    if (!processed) return null;
    const { centers, counts, size } = processed.hist;
    const pos = { x: [], y: [] };
    const neg = { x: [], y: [] };
    centers.forEach((c, i) => {
      const t = c >= 0 ? pos : neg;
      t.x.push(c);
      t.y.push(counts[i]);
    });
    const bar = (t, name, color) => ({
      x: t.x, y: t.y, type: "bar", name, width: size * 0.95, marker: { color },
    });
    return {
      data: [
        bar(neg, "Predicted to lose buyers", "#d64545"),
        bar(pos, "Predicted to gain buyers", "#2f6fed"),
      ],
      layout: {
        title: "Predicted effect of a $" + level + " discount",
        xaxis: { title: "Predicted change in purchase probability", tickformat: ".0%" },
        yaxis: { title: "Customers" },
        legend: { orientation: "h", y: -0.3 },
      },
    };
  }, [processed, level]);

  const scatterChart = useMemo(() => {
    if (!processed) return null;
    const rows = processed.rows;
    return {
      data: [{
        x: rows.map((r) => r.days_since_last_purchase),
        y: rows.map((r) => r[key]),
        type: "scattergl",
        mode: "markers",
        marker: {
          size: 4,
          opacity: 0.5,
          color: rows.map((r) => r[key]),
          colorscale: [[0, "#d64545"], [0.5, "#d9dee4"], [1, "#2f6fed"]],
          cmid: 0,
        },
        hovertemplate:
          "Days since last purchase: %{x:.0f}<br>Predicted effect: %{y:.1%}<extra></extra>",
      }],
      layout: {
        title: "Predicted effect vs days since last purchase ($" + level + ")",
        xaxis: { title: "Days since last purchase" },
        yaxis: { title: "Predicted change in purchase probability", tickformat: ".0%" },
      },
    };
  }, [processed, key, level]);

  const observed = useMemo(() => {
    if (!dataset) return null;
    const r = purchaseRateByDiscount(dataset.rows);
    return {
      data: [{
        x: r.map((v) => "$" + v.level),
        y: r.map((v) => v.rate),
        type: "bar",
        marker: { color: ["#9aa5b1", "#2f6fed", "#1f4fb8"] },
        text: r.map((v) => (v.rate * 100).toFixed(1) + "% (n=" + v.n + ")"),
        textposition: "outside",
      }],
      layout: {
        title: "Observed purchase rate by discount (not causal)",
        xaxis: { title: "Discount received" },
        yaxis: { title: "Purchase rate", tickformat: ".0%", rangemode: "tozero" },
      },
    };
  }, [dataset]);

  if (error) {
    return (
      <div className="card">
        <p className="error">{error}</p>
        <p>Generate the demo files from the project root:</p>
        <pre>
          python backend\src\model_comparison.py{"\n"}
          python backend\src\export_demo.py
        </pre>
      </div>
    );
  }

  if (!qini || !customers) {
    return (
      <div className="card">
        <p>Loading demo data...</p>
      </div>
    );
  }

  const s = processed.stats;

  return (
    <div>
      <div className="banner">
        Qini curves use held-out synthetic customers and their known true
        effects. The predicted effects below are Double ML output for 10,000
        synthetic customers.
      </div>

      <div className="card">
        <PlotlyChart data={qiniChart.data} layout={qiniChart.layout} height={460} />
        <p>
          Selected model: <strong>{qini.selected_model}</strong>. A curve above
          the dashed line beats a random rollout.
        </p>
      </div>

      <div className="card">
        <h2>Filter customers</h2>
        <div className="filters">
          <div>
            <label>Discount level</label>
            <select value={level} onChange={(e) => setLevel(Number(e.target.value))}>
              <option value={10}>$10</option>
              <option value={20}>$20</option>
            </select>
          </div>
          <div>
            <label>Loyalty status</label>
            <select
              value={filters.loyalty}
              onChange={(e) => setFilters((f) => ({ ...f, loyalty: e.target.value }))}
            >
              <option value="all">All customers</option>
              <option value="members">Loyalty members</option>
              <option value="non">Non-members</option>
            </select>
          </div>
          <div>
            <label>Days since last purchase, minimum: {filters.recencyMin}</label>
            <input
              type="range" min="0" max="365" value={filters.recencyMin}
              onChange={(e) => {
                const v = Number(e.target.value);
                setFilters((f) => ({ ...f, recencyMin: Math.min(v, f.recencyMax) }));
              }}
            />
          </div>
          <div>
            <label>Days since last purchase, maximum: {filters.recencyMax}</label>
            <input
              type="range" min="0" max="365" value={filters.recencyMax}
              onChange={(e) => {
                const v = Number(e.target.value);
                setFilters((f) => ({ ...f, recencyMax: Math.max(v, f.recencyMin) }));
              }}
            />
          </div>
        </div>
        <div className="checks">
          <label>
            <input
              type="checkbox" checked={filters.heldOutOnly}
              onChange={() => setFilters((f) => ({ ...f, heldOutOnly: !f.heldOutOnly }))}
            />{" "}
            Held-out customers only (never seen by the model)
          </label>
        </div>

        <div className="stats">
          <div className="stat"><strong>{s.n.toLocaleString()}</strong>Customers shown</div>
          <div className="stat"><strong>{points(s.mean)}</strong>Average predicted effect</div>
          <div className="stat">
            <strong>{(s.positiveShare * 100).toFixed(1)}%</strong>Predicted to gain buyers
          </div>
          <div className="stat">
            <strong>{processed.ms.toFixed(1)} ms</strong>Filter and summarize
          </div>
        </div>
        <p className="small">
          Loaded {customers.length.toLocaleString()} customers in {loadMs} ms.
        </p>
      </div>

      <div className="card">
        <PlotlyChart data={histChart.data} layout={histChart.layout} />
      </div>

      <div className="card">
        <PlotlyChart data={scatterChart.data} layout={scatterChart.layout} height={420} />
      </div>

      <div className="card">
        {observed ? (
          <>
            <PlotlyChart data={observed.data} layout={observed.layout} />
            <p>
              Raw purchase rates from your uploaded file. Discounts went mostly
              to inactive customers, so this comparison is biased, which is why
              the causal estimates above differ from it.
            </p>
          </>
        ) : (
          <p>Upload a campaign CSV on the first tab to compare with the raw purchase rates.</p>
        )}
      </div>
    </div>
  );
}