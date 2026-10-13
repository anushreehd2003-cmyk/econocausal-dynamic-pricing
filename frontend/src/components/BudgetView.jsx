export default function BudgetView({ settings, setSettings, dataset }) {
  const budgetValid = Number(settings.budget) > 0;
  const budget = Number(settings.budget);

  function toggle(level) {
    setSettings({
      ...settings,
      allowed: { ...settings.allowed, [level]: !settings.allowed[level] },
    });
  }

  const noLevel = !settings.allowed[10] && !settings.allowed[20];

  return (
    <div className="card">
      <h2>Set budget constraints</h2>

      <label>Total marketing budget ($)</label>
      <input
        type="number"
        min="1"
        value={settings.budget}
        onChange={(e) => setSettings({ ...settings, budget: e.target.value })}
      />
      {!budgetValid && <p className="error">Enter a budget greater than 0.</p>}

      <label>Allowed discount levels</label>
      <div className="checks">
        <label>
          <input type="checkbox" checked={settings.allowed[10]} onChange={() => toggle(10)} /> $10
        </label>
        <label>
          <input type="checkbox" checked={settings.allowed[20]} onChange={() => toggle(20)} /> $20
        </label>
      </div>
      {noLevel && <p className="error">Select at least one discount level.</p>}

      {budgetValid && !noLevel && (
        <div className="stats">
          {settings.allowed[10] && (
            <div className="stat">
              <strong>{Math.floor(budget / 10)}</strong>customers at $10
            </div>
          )}
          {settings.allowed[20] && (
            <div className="stat">
              <strong>{Math.floor(budget / 20)}</strong>customers at $20
            </div>
          )}
          {dataset && (
            <div className="stat">
              <strong>{dataset.rows.length}</strong>customers in data
            </div>
          )}
        </div>
      )}

      <p>
        These settings feed the SciPy optimizer in Week 3. The allocation
        decides who actually receives a discount.
      </p>
    </div>
  );
}