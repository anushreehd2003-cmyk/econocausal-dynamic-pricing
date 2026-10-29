// Greedy budget allocation: a JavaScript port of greedy_allocation in
// backend/src/optimizer.py. Gives each customer at most one discount, best
// predicted revenue per budget dollar first, never exceeding the budget.

export const LEVELS = [10, 20];

// customers: [{ ite_10, ite_20, avg_basket_value }, ...]
// returns an array of discounts (0, 10 or 20), one per customer
export function greedyAllocation(customers, budget, levels = LEVELS) {
  const candidates = [];
  customers.forEach((c, i) => {
    levels.forEach((level) => {
      const value = c["ite_" + level] * c.avg_basket_value;
      if (value > 0) candidates.push({ i, level, ratio: value / level });
    });
  });
  // Array.prototype.sort is stable, so ties keep their original order,
  // matching numpy's stable argsort in the Python version.
  candidates.sort((a, b) => b.ratio - a.ratio);

  const allocation = new Array(customers.length).fill(0);
  let left = budget;
  for (const c of candidates) {
    if (allocation[c.i] === 0 && c.level <= left) {
      allocation[c.i] = c.level;
      left -= c.level;
    }
  }
  return allocation;
}

export function summarizeAllocation(customers, allocation) {
  let spend = 0;
  let targeted = 0;
  let extraPurchases = 0;
  let extraRevenue = 0;
  let n10 = 0;
  let n20 = 0;
  customers.forEach((c, i) => {
    const level = allocation[i];
    if (level === 0) return;
    const ite = c["ite_" + level];
    spend += level;
    targeted += 1;
    extraPurchases += ite;
    extraRevenue += ite * c.avg_basket_value;
    if (level === 10) n10 += 1;
    if (level === 20) n20 += 1;
  });
  return { spend, targeted, n10, n20, extraPurchases, extraRevenue };
}