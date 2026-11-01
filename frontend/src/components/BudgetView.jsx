import { useMemo } from "react";
import { LEVELS, greedyAllocation, summarizeAllocation } from "../utils/allocation";

export default function BudgetView({ settings, setSettings, dataset, customers }) {
  const budget = Number(settings.budget);
  const budgetValid = budget > 0;
  const useTen = Boolean(settings.allowed[10]);
  const useTwenty = Boolean(settings.allowed[20]);
  const noLevel = !useTen && !useTwenty;

  const preview = useMemo(() => {
    if (!customers || !(budget > 0) || (!useTen && !useTwenty)) return null;
    const levels = LEVELS.filter((l) => (l === 10 ? useTen : useTwenty));
    return summarizeAllocation(customers, greedyAllocation(customers, budget, levels));
  }, [customers, budget, useTen, useTwenty]);

  function toggle(level) {
    setSettings({
      ...settings,
      allowed: { ...settings.allowed, [level]: !settings.allowed[level] },
    });
  }

  return (
    <div className="card">
      <h2>Set budget constraints</h2>

      <label htmlFor="budget-input">Total marketing budget ($)</label>
      <input
        id="budget-input"
        type="number"
        min="1"
        value={settings.budget}
        onChange={(e) => setSettings({ ...settings, budget: e.target.value })}
      />
      {!budgetValid && <p className="error">Enter a budget greater than 0.</p>}

      <label>Allowed discount levels</label>
      <div className="checks">
        <label>
          <input type="checkbox" checked={useTen} onChange={() => toggle(10)} /> $10
        </label>
        <label>
          <input type="checkbox" checked={useTwenty} onChange={() => toggle(20)} /> $20
        </label>
      </div>
      {noLevel && <p className="error">Select at least one discount level.</p>}

      {budgetValid && !noLevel && (
        <div className="stats">
          {useTen && (
            <div className="stat">
              <strong>{Math.floor(budget / 10)}</strong>most customers at $10
            </div>
          )}
          {useTwenty && (
            <div className="stat">
              <strong>{Math.floor(budget / 20)}</strong>most customers at $20
            </div>
          )}
          {dataset && (
            <div className="stat">
              <strong>{dataset.rows.length}</strong>customers in uploaded data
            </div>
          )}
        </div>
      )}

      {preview && (
        <div className="summary">
          <strong>Plan preview:</strong> this budget reaches{" "}
          {preview.targeted.toLocaleString()} customers ({preview.n10} at $10,{" "}
          {preview.n20} at $20), spends ${preview.spend.toLocaleString()} and is
          predicted to add about ${Math.round(preview.extraRevenue).toLocaleString()} in
          gross revenue. Open the Allocation tab for the full prescription.
        </div>
      )}

      <p>
        These settings drive the allocation: the plan decides who actually
        receives a discount and how much.
      </p>
    </div>
  );
}