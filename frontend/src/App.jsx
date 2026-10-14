import { useState } from "react";
import "./App.css";
import UploadView from "./components/UploadView";
import BudgetView from "./components/BudgetView";
import UpliftView from "./components/UpliftView";

export default function App() {
  const [tab, setTab] = useState("upload");
  const [dataset, setDataset] = useState(null);
  const [settings, setSettings] = useState({
    budget: 5000,
    allowed: { 10: true, 20: true },
  });

  return (
    <div className="app">
      <div className="header">
        <h1>EconoCausal</h1>
        <p>Causal uplift and budget allocation dashboard</p>
      </div>

      <div className="tabs">
        <button className={tab === "upload" ? "active" : ""} onClick={() => setTab("upload")}>
          1. Upload data
        </button>
        <button className={tab === "budget" ? "active" : ""} onClick={() => setTab("budget")}>
          2. Budget
        </button>
        <button className={tab === "uplift" ? "active" : ""} onClick={() => setTab("uplift")}>
          3. Uplift
        </button>
      </div>

      {tab === "upload" && <UploadView dataset={dataset} setDataset={setDataset} />}
      {tab === "budget" && (
        <BudgetView settings={settings} setSettings={setSettings} dataset={dataset} />
      )}
      {tab === "uplift" && <UpliftView dataset={dataset} />}
    </div>
  );
}