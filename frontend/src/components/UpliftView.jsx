import { useMemo } from "react";
import PlotlyChart from "./PlotlyChart";
import {
  placeholderQini,
  placeholderDecileUplift,
  purchaseRateByDiscount,
} from "../utils/placeholderCurves";

export default function UpliftView({ dataset }) {
  const qini = useMemo(() => {
    const q = placeholderQini();
    return {
      data: [
        { x: q.x, y: q.randomPolicy, type: "scatter", mode: "lines",
          name: "Random targeting", line: { dash: "dash", color: "#9aa5b1" } },
        { x: q.x, y: q.demoModel, type: "scatter", mode: "lines",
          name: "Causal model (demo)", line: { color: "#2f6fed", width: 3 } },
      ],
      layout: {
        title: "Qini curve (placeholder)",
        xaxis: { title: "% of customers targeted" },
        yaxis: { title: "Cumulative incremental purchases" },
        legend: { orientation: "h", y: -0.25 },
      },
    };
  }, []);

  const decile = useMemo(() => {
    const d = placeholderDecileUplift();
    return {
      data: [{
        x: d.deciles, y: d.uplift, type: "bar",
        marker: { color: d.uplift.map((v) => (v >= 0 ? "#2f6fed" : "#d64545")) },
      }],
      layout: {
        title: "Uplift by decile (placeholder)",
        xaxis: { title: "Decile (1 = highest predicted uplift)" },
        yaxis: { title: "Average uplift in purchase probability", tickformat: ".0%" },
      },
    };
  }, []);

  const observed = useMemo(() => {
    if (!dataset) return null;
    const r = purchaseRateByDiscount(dataset.rows);
    return {
      stats: r,
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

  return (
    <div>
      <div className="banner">
        Preview: the two curves below are placeholder shapes. In Week 2 they
        are replaced by the real Double ML model output.
      </div>

      <div className="card">
        <PlotlyChart data={qini.data} layout={qini.layout} />
      </div>

      <div className="card">
        <PlotlyChart data={decile.data} layout={decile.layout} />
      </div>

      <div className="card">
        {observed ? (
          <>
            <PlotlyChart data={observed.data} layout={observed.layout} />
            <p>
              This compares raw purchase rates. Discounts were given mostly to
              inactive customers, so this comparison is biased. The causal
              model corrects for that.
            </p>
          </>
        ) : (
          <p>Upload a campaign CSV on the first tab to see observed purchase rates here.</p>
        )}
      </div>
    </div>
  );
}