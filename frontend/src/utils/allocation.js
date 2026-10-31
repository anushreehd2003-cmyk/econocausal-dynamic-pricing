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

// Expected result of giving the lowest allowed discount to randomly chosen
// customers until the budget runs out (the "blanket targeting" baseline).
export function blanketExpected(customers, budget, levels = LEVELS) {
  const level = Math.min(...levels);
  const targeted = Math.min(customers.length, Math.floor(budget / level));
  let sumEffect = 0;
  let sumRevenue = 0;
  for (const c of customers) {
    sumEffect += c["ite_" + level];
    sumRevenue += c["ite_" + level] * c.avg_basket_value;
  }
  return {
    level,
    targeted,
    extraPurchases: (sumEffect / customers.length) * targeted,
    extraRevenue: (sumRevenue / customers.length) * targeted,
  };
}

// One row per customer for the prescription table, best predicted revenue
// first. Customers with no discount have a predicted extra revenue of 0.
export function prescriptionRows(customers, allocation) {
  const rows = customers.map((c, i) => {
    const level = allocation[i];
    const effect = level === 0 ? 0 : c["ite_" + level];
    return {
      customer_id: c.customer_id,
      discount: level,
      ite_10: c.ite_10,
      ite_20: c.ite_20,
      predicted_effect: effect,
      predicted_extra_revenue: effect * c.avg_basket_value,
      days_since_last_purchase: c.days_since_last_purchase,
      loyalty_member: c.loyalty_member,
    };
  });
  rows.sort((a, b) => b.predicted_extra_revenue - a.predicted_extra_revenue);
  return rows;
}