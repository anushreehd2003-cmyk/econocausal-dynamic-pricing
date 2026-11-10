import test from "node:test";
import assert from "node:assert/strict";
import { toDriftPayload, MAX_ROWS } from "../src/utils/drift.js";

const good = (over = {}) => ({
  customer_id: 1, age: 35, income: 52000, loyalty_member: 0,
  past_purchases: 2, days_since_last_purchase: 120,
  avg_basket_value: 60, discount: 10, purchased: 1, revenue: 55, ...over,
});

test("keeps the features and outcomes and drops unrelated columns", () => {
  const out = toDriftPayload([good()]);
  assert.equal(out.customers.length, 1);
  const c = out.customers[0];
  assert.deepEqual(Object.keys(c).sort(), [
    "age", "customer_id", "days_since_last_purchase", "discount", "income",
    "loyalty_member", "past_purchases", "purchased",
  ]);
  assert.equal(c.discount, 10);
  assert.equal(c.purchased, 1);
  assert.equal("revenue" in c, false);
  assert.equal("avg_basket_value" in c, false);
});

test("skips rows with missing or invalid features and counts them", () => {
  const rows = [
    good(),
    good({ age: null }),
    good({ income: "n/a" }),
    good({ loyalty_member: 2 }),
    good({ age: 200 }),
    good({ days_since_last_purchase: -1 }),
    good({ past_purchases: NaN }),
  ];
  const out = toDriftPayload(rows);
  assert.equal(out.customers.length, 1);
  assert.equal(out.skipped, 6);
  assert.equal(out.total, 7);
});

test("invalid outcome values become null instead of dropping the row", () => {
  const out = toDriftPayload([good({ discount: 15, purchased: 3 }), good({ discount: undefined, purchased: undefined })]);
  assert.equal(out.customers.length, 2);
  assert.equal(out.customers[0].discount, null);
  assert.equal(out.customers[0].purchased, null);
  assert.equal(out.customers[1].discount, null);
});

test("sends at most MAX_ROWS customers and reports truncation", () => {
  const rows = Array.from({ length: MAX_ROWS + 5 }, (_, i) => good({ customer_id: i }));
  const out = toDriftPayload(rows);
  assert.equal(out.customers.length, MAX_ROWS);
  assert.equal(out.truncated, true);
  assert.equal(toDriftPayload([good()]).truncated, false);
});