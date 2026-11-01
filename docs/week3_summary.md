# Week 3 Summary

## Delivered
- Budget optimizer with exact integer program, greedy and blanket baselines (backend/src/optimizer.py)
- Held-out evaluation against an oracle that knows the true effects
- Budget-efficiency curve for nine budgets (backend/src/efficiency.py)
- JavaScript greedy allocator with a parity test against the Python version (frontend/src/utils/allocation.js)
- Dashboard Allocation tab (prescription table with search, filter, sort and CSV export, plus the efficiency chart) and a plan preview on the Budget tab
- 26 Python unit tests and JavaScript tests for the allocator

## Key results
| Strategy (3,000 held-out customers, $1,500 budget) | True extra revenue | Extra purchases |
|---|---|---|
| Optimizer (exact) | $1,667 | 15.95 |
| Greedy | $1,687 | 16.15 |
| Blanket $10 (random) | $206 | 3.63 |
| Oracle (true effects) | $2,128 | 22.24 |

- Optimized targeting earns 4.4 to 8.1 times the extra revenue of random blanket discounts for budgets from $500 to $20,000 (per 10,000 customers)
- Greedy is within about 1% of the exact optimizer on true revenue, so the live dashboard uses it
- Returns diminish: each extra budget dollar returns about 1.1 to 1.25 dollars of gross revenue up to $3,000, and 0.47 between $15,000 and $20,000
- Almost the whole budget goes to $10 discounts, because $20 rarely gives more than twice the effect

## Limitations
- Revenue is gross and ignores margin given to customers who would have bought anyway
- True effects exist only because the data is synthetic
- Dashboard predictions are in-sample for about 70% of customers

## Next (Week 4)
REST API, data drift detector, natural-language summaries and final polish.