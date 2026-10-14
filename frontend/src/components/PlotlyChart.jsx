import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export default function PlotlyChart({ data, layout, height = 380 }) {
  const ref = useRef(null);

  useEffect(() => {
    Plotly.react(
      ref.current,
      data,
      { autosize: true, margin: { t: 50, l: 60, r: 20, b: 60 }, ...layout },
      { responsive: true, displaylogo: false }
    );
  }, [data, layout]);

  useEffect(() => {
    const el = ref.current;
    return () => Plotly.purge(el);
  }, []);

  return <div ref={ref} style={{ width: "100%", height }} />;
}