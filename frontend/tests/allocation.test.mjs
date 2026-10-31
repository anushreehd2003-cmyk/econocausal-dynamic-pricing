import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";


import {
  greedyAllocation,
  summarizeAllocation,
  blanketExpected,
  prescriptionRows,
} from "../src/utils/allocation.js";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/greedy_parity.json", import.meta.url), "utf8")
);

test("JavaScript greedy matches the Python greedy allocation", () => {
  assert.ok(fixture.cases.length > 0);
  for (const c of fixture.cases) {
    const got = greedyAllocation(fixture.customers, c.budget, c.levels);
    assert.deepEqual(got, c.allocation, "budget " + c.budget + ", levels " + c.levels);
  }
});

test("never spends more than the budget", () => {
  for (const budget of [0, 5, 10, 95, 500, 5000]) {
    const alloc = greedyAllocation(fixture.customers, budget);
    const spend = alloc.reduce((a, b) => a + b, 0);
    assert.ok(spend <= budget);
  }
});

test("only allowed levels are used", () => {
  const alloc = greedyAllocation(fixture.customers, 5000, [10]);
  assert.ok(alloc.every((v) => v === 0 || v === 10));
});

test("summary matches a hand calculation", () => {
  const customers = [
    { ite_10: 0.05, ite_20: 0.1, avg_basket_value: 100 },
    { ite_10: 0.02, ite_20: -0.03, avg_basket_value: 50 },
    { ite_10: 0.04, ite_20: 0.08, avg_basket_value: 10 },
  ];
  const s = summarizeAllocation(customers, [20, 10, 0]);
  assert.equal(s.spend, 30);
  assert.equal(s.targeted, 2);
  assert.equal(s.n10, 1);
  assert.equal(s.n20, 1);
  assert.ok(Math.abs(s.extraPurchases - 0.12) < 1e-12);
  assert.ok(Math.abs(s.extraRevenue - 11) < 1e-12);
});


test("blanket expectation matches a hand calculation", () => {
  const customers = [
    { ite_10: 0.1, ite_20: 0.2, avg_basket_value: 100 },
    { ite_10: 0.3, ite_20: 0.0, avg_basket_value: 100 },
  ];
  const a = blanketExpected(customers, 10);
  assert.equal(a.level, 10);
  assert.equal(a.targeted, 1);
  assert.ok(Math.abs(a.extraRevenue - 20) < 1e-9);
  const b = blanketExpected(customers, 100);
  assert.equal(b.targeted, 2);
  assert.ok(Math.abs(b.extraRevenue - 40) < 1e-9);
  const c = blanketExpected(customers, 40, [20]);
  assert.equal(c.level, 20);
  assert.equal(c.targeted, 2);
  assert.ok(Math.abs(c.extraRevenue - 20) < 1e-9);
});

test("prescription rows carry the chosen discount and its effect", () => {
  const customers = [
    { customer_id: 8, ite_10: -0.02, ite_20: -0.03, avg_basket_value: 50,
      days_since_last_purchase: 40, loyalty_member: 0 },
    { customer_id: 7, ite_10: 0.05, ite_20: 0.1, avg_basket_value: 100,
      days_since_last_purchase: 12, loyalty_member: 1 },
  ];
  const rows = prescriptionRows(customers, [0, 20]);
  assert.equal(rows[0].customer_id, 7); // best predicted revenue first
  assert.equal(rows[0].discount, 20);
  assert.ok(Math.abs(rows[0].predicted_effect - 0.1) < 1e-12);
  assert.ok(Math.abs(rows[0].predicted_extra_revenue - 10) < 1e-12);
  assert.equal(rows[1].discount, 0);
  assert.equal(rows[1].predicted_effect, 0);
  assert.equal(rows[1].predicted_extra_revenue, 0);
});