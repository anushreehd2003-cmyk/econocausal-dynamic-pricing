import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import PlotlyChart from "./PlotlyChart";
import { loadEfficiency } from "../utils/demoData";
import {
  LEVELS,
  greedyAllocation,
  summarizeAllocation,
  blanketExpected,
  prescriptionRows,
} from "../utils/allocation";

const PAGE_SIZE = 20;

const VIEWS = {
  targeted: { label: "Customers with a discount", test: (r) => r.discount > 0 },
  d10: { label: "$10 discounts only", test: (r) => r.discount === 10 },
  d20: { label: "$20 discounts only", test: (r) => r.discount === 20 },
  none: { label: "Customers with no discount", test: (r) => r.discount === 0 },
  all: { label: "All customers", test: () => true },
};

const SORTS = {
  revenue: {
    label: "Predicted extra revenue (high to low)",
    cmp: (a, b) => b.predicted_extra_revenue - a.predicted_extra_revenue,
  },
  effect10: {
    label: "Effect of $10 (high to low)",
    cmp: (a, b) => b.ite_10 - a.ite_10,
  },
  effect20: {
    label: "Effect of $20 (high to low)",
    cmp: (a, b) => b.ite_20 - a.ite_20,
  },
  recency: {
    label: "Days since last purchase (most recent first)",
    cmp: (a, b) => a.days_since_last_purchase - b.days_since_last_purchase,
  },
  id: {
    label: "Customer ID (low to high)",
    cmp: (a, b) => a.customer_id - b.customer_id,
  },
};

function money(value) {
  return "$" + Math.round(value).toLocaleString();
}

function pts(value) {
  const p = value * 100;
  return (p >= 0 ? "+" : "") + p.toFixed(1) + " pts";
}

export default function AllocationView({ settings, customers, customersError }) {
  const [efficiency, setEfficiency] = useState(null);
  const [error, setError] = useState("");
  const [view, setView] = useState("targeted");
  const [sort, setSort] = useState("revenue");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);

  useEffect(() => {
    let cancelled = false;
    loadEfficiency()
      .then((e) => {
        if (!cancelled) setEfficiency(e);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const budget = Number(settings.budget);
  const useTen = Boolean(settings.allowed[10]);
  const useTwenty = Boolean(settings.allowed[20]);
  const valid = budget > 0 && (useTen || useTwenty);

  const plan = useMemo(() => {
    if (!customers || !valid) return null;
    const levels = LEVELS.filter((l) => (l === 10 ? useTen : useTwenty));
    const allocation = greedyAllocation(customers, budget, levels);
    return {
      summary: summarizeAllocation(customers, allocation),
      blanket: blanketExpected(customers, budget, levels),
      rows: prescriptionRows(customers, allocation),
    };
  }, [customers, valid, budget, useTen, useTwenty]);

  const visible = useMemo(() => {
    if (!plan) return [];
    const q = query.trim();
    const test = VIEWS[view].test;
    const rows = plan.rows.filter(
      (r) => test(r) && (!q || String(r.customer_id).includes(q))
    );
    rows.sort(SORTS[sort].cmp);
    return rows;
  }, [plan, view, sort, query]);

  const pageCount = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  const current = Math.min(page, pageCount - 1);
  const shown = visible.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);

  const chart = useMemo(() => {
    if (!efficiency) return null;
    const style = {
      oracle: ["Oracle (true effects)", { dash: "dot", color: "#1f2933" }],
      optimizer: ["Optimizer (exact)", { color: "#2f6fed", width: 3 }],
      greedy: ["Greedy (live allocation)", { dash: "dash", color: "#12a594" }],
      blanket: ["Blanket $10 (random)", { color: "#9aa5b1" }],
    };
    const data = Object.keys(style).map((name) => ({
      x: efficiency.budget,
      y: efficiency.series[name].revenue,
      type: "scatter",
      mode: "lines+markers",
      name: style[name][0],
      line: style[name][1],
    }));
    const marker = {
      type: "line", xref: "x", yref: "paper",
      x0: budget, x1: budget, y0: 0, y1: 1,
      line: { color: "#d64545", width: 2, dash: "dot" },
    };
    return {
      data,
      layout: {
        title: "Budget efficiency (held-out customers, true effects)",
        xaxis: { title: "Budget ($, per 10,000 customers)" },
        yaxis: { title: "Extra revenue ($, per 10,000 customers)" },
        legend: { orientation: "h", y: -0.3 },
        shapes: valid ? [marker] : [],
      },
    };
  }, [efficiency, valid, budget]);

  function downloadCsv() {
    const rows = plan.rows
      .filter((r) => r.discount > 0)
      .map((r) => ({
        customer_id: r.customer_id,
        discount: r.discount,
        predicted_effect: Number(r.predicted_effect.toFixed(4)),
        predicted_extra_revenue: Number(r.predicted_extra_revenue.toFixed(2)),
        days_since_last_purchase: Math.round(r.days_since_last_purchase),
        loyalty_member: r.loyalty_member,
      }));
    const blob = new Blob([Papa.unparse(rows)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "prescription_budget_" + budget + ".csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  const loadError = customersError || error;
  if (loadError) {
    return (
      <div className="card">
        <p className="error">{loadError}</p>
        <p>Generate the demo files from the project root:</p>
        <pre>
          python backend\src\export_demo.py{"\n"}
          python backend\src\efficiency.py
        </pre>
      </div>
    );
  }

  if (!customers || !efficiency) {
    return (
      <div className="card">
        <p>Loading allocation data...</p>
      </div>
    );
  }

  if (!valid) {
    return (
      <div className="card">
        <h2>Prescription</h2>
        <p className="error">
          Set a budget above 0 and allow at least one discount level on the
          Budget tab.
        </p>
      </div>
    );
  }

  const s = plan.summary;
  const ratio =
    plan.blanket.extraRevenue > 0 ? s.extraRevenue / plan.blanket.extraRevenue : null;

  return (
    <div>
      <div className="card">
        <h2>Prescription</h2>
        <p className="summary">
          With a {money(budget)} budget, the plan gives discounts to{" "}
          {s.targeted.toLocaleString()} customers ({s.n10} at $10, {s.n20} at $20)
          and is predicted to add about {s.extraPurchases.toFixed(0)} purchases and{" "}
          {money(s.extraRevenue)} in gross revenue
          {ratio
            ? ", about " + ratio.toFixed(1) + " times what the same budget earns when spread over random customers."
            : "."}
        </p>
        <div className="stats">
          <div className="stat"><strong>{money(budget)}</strong>Budget</div>
          <div className="stat"><strong>{money(s.spend)}</strong>Spent</div>
          <div className="stat"><strong>{s.targeted.toLocaleString()}</strong>Customers targeted</div>
          <div className="stat"><strong>{s.extraPurchases.toFixed(1)}</strong>Predicted extra purchases</div>
          <div className="stat"><strong>{money(s.extraRevenue)}</strong>Predicted extra revenue (gross)</div>
          <div className="stat">
            <strong>{money(plan.blanket.extraRevenue)}</strong>
            Blanket ${plan.blanket.level}, random (expected)
          </div>
        </div>
        <p className="small">
          Live allocation uses a fast greedy method. Predicted values come from
          Double ML and are in-sample for about 70% of these customers, so treat
          totals as a demo. Revenue is gross: it ignores margin given to
          customers who would have bought anyway.
        </p>
      </div>

      <div className="card">
        <div className="toolbar">
          <div>
            <label htmlFor="find-customer">Find customer ID</label>
            <input
              id="find-customer"
              type="text"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(0);
              }}
            />
          </div>
          <div>
            <label htmlFor="view-select">Show</label>
            <select
              id="view-select"
              value={view}
              onChange={(e) => {
                setView(e.target.value);
                setPage(0);
              }}
            >
              {Object.keys(VIEWS).map((key) => (
                <option
                  key={key}
                  value={key}
                  disabled={(key === "d10" && !useTen) || (key === "d20" && !useTwenty)}
                >
                  {VIEWS[key].label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="sort-select">Sort by</label>
            <select
              id="sort-select"
              value={sort}
              onChange={(e) => {
                setSort(e.target.value);
                setPage(0);
              }}
            >
              {Object.keys(SORTS).map((key) => (
                <option key={key} value={key}>
                  {SORTS[key].label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <button className="primary" onClick={downloadCsv}>
              Download prescription (CSV)
            </button>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Customer</th>
                <th>Prescribed discount</th>
                <th className="num">Effect of $10</th>
                <th className="num">Effect of $20</th>
                <th className="num">Predicted extra revenue</th>
                <th className="num">Days since last purchase</th>
                <th>Loyalty member</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.customer_id}>
                  <td>{r.customer_id}</td>
                  <td>
                    <span className={"pill d" + r.discount}>
                      {r.discount === 0 ? "None" : "$" + r.discount}
                    </span>
                  </td>
                  <td className="num">{pts(r.ite_10)}</td>
                  <td className="num">{pts(r.ite_20)}</td>
                  <td className="num">
                    {r.discount === 0 ? "-" : "$" + r.predicted_extra_revenue.toFixed(2)}
                  </td>
                  <td className="num">{Math.round(r.days_since_last_purchase)}</td>
                  <td>{r.loyalty_member === 1 ? "Yes" : "No"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {shown.length === 0 && <p>No customers match.</p>}

        <div className="pager">
          <button disabled={current === 0} onClick={() => setPage(current - 1)}>
            Previous
          </button>
          <span>
            Page {current + 1} of {pageCount} ({visible.length.toLocaleString()} customers)
          </span>
          <button
            disabled={current >= pageCount - 1}
            onClick={() => setPage(current + 1)}
          >
            Next
          </button>
        </div>
      </div>

      <div className="card">
        <PlotlyChart data={chart.data} layout={chart.layout} height={440} />
        <p>
          The red dotted line marks your current budget. This chart uses
          held-out synthetic customers and their true effects, scaled to 10,000
          customers, so it is a fairer test than the predicted totals above.
        </p>
      </div>
    </div>
  );
}