import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { greedyAllocation, summarizeAllocation } from "../src/utils/allocation.js";

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