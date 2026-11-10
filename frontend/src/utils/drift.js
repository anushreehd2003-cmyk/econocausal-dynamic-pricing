// Turns uploaded campaign rows into a valid request for POST /drift.

export const FEATURES = [
  "age",
  "income",
  "loyalty_member",
  "past_purchases",
  "days_since_last_purchase",
];
export const MAX_ROWS = 10000; // the API accepts at most this many customers
export const MIN_ROWS = 100; // the API needs at least this many for a drift check

function isNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function validFeatures(r) {
  return (
    FEATURES.every((f) => isNumber(r[f])) &&
    r.age >= 0 && r.age <= 120 &&
    r.income >= 0 &&
    (r.loyalty_member === 0 || r.loyalty_member === 1) &&
    r.past_purchases >= 0 &&
    r.days_since_last_purchase >= 0
  );
}

export function toDriftPayload(rows) {
  const customers = [];
  let skipped = 0;
  let valid = 0;
  for (const r of rows) {
    if (!validFeatures(r)) {
      skipped += 1;
      continue;
    }
    valid += 1;
    if (customers.length >= MAX_ROWS) continue;
    const c = {};
    if (r.customer_id !== undefined && r.customer_id !== null) c.customer_id = r.customer_id;
    for (const f of FEATURES) c[f] = r[f];
    // Outcome columns are optional: send them only when they are valid
    c.discount = [0, 10, 20].includes(r.discount) ? r.discount : null;
    c.purchased = r.purchased === 0 || r.purchased === 1 ? r.purchased : null;
    customers.push(c);
  }
  return {
    customers,
    total: rows.length,
    skipped,
    truncated: valid > MAX_ROWS,
  };
}