export function filterCustomers(rows, f) {
  return rows.filter((r) => {
    if (f.heldOutOnly && r.is_test !== 1) return false;
    if (f.loyalty === "members" && r.loyalty_member !== 1) return false;
    if (f.loyalty === "non" && r.loyalty_member !== 0) return false;
    const d = r.days_since_last_purchase;
    return d >= f.recencyMin && d <= f.recencyMax;
  });
}

export function summarize(rows, key) {
  if (rows.length === 0) {
    return { n: 0, mean: 0, positive: 0, positiveShare: 0 };
  }
  let sum = 0;
  let positive = 0;
  for (const r of rows) {
    sum += r[key];
    if (r[key] > 0) positive += 1;
  }
  return {
    n: rows.length,
    mean: sum / rows.length,
    positive,
    positiveShare: positive / rows.length,
  };
}

export function histogram(values, lo, hi, bins) {
  const size = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let i = Math.floor((v - lo) / size);
    if (i < 0) i = 0;
    if (i >= bins) i = bins - 1;
    counts[i] += 1;
  }
  const centers = counts.map((_, i) => lo + size * (i + 0.5));
  return { centers, counts, size };
}