// PLACEHOLDER data only. Replaced in Week 2 by real Double ML output.

export function placeholderQini(totalIncremental = 100, points = 21) {
  const x = [];
  const randomPolicy = [];
  const demoModel = [];
  for (let i = 0; i < points; i++) {
    const f = i / (points - 1); // fraction of customers targeted
    x.push(Number((f * 100).toFixed(0)));
    randomPolicy.push(Number((totalIncremental * f).toFixed(2)));
    // rises fast, then flattens and dips slightly (sleeping dogs at the end)
    demoModel.push(Number((totalIncremental * (2.2 * f - 1.2 * f * f)).toFixed(2)));
  }
  return { x, randomPolicy, demoModel };
}

export function placeholderDecileUplift() {
  return {
    deciles: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    uplift: [0.18, 0.14, 0.11, 0.08, 0.05, 0.03, 0.0, -0.02, -0.04, -0.06],
  };
}

// REAL data: observed purchase rate per discount level from the upload.
export function purchaseRateByDiscount(rows) {
  const levels = [0, 10, 20];
  return levels.map((d) => {
    const group = rows.filter((r) => r.discount === d);
    const buyers = group.filter((r) => r.purchased === 1).length;
    return {
      level: d,
      n: group.length,
      rate: group.length ? buyers / group.length : 0,
    };
  });
}