# Data Drift Detection (Week 4, Day 20)

Warns when new customers or their behaviour differ from the data the model
was trained on. Code: `backend/src/drift.py`, demo: `backend/src/drift_demo.py`,
API: `POST /drift`.

| Check | Compares | Method |
|---|---|---|
| Input drift | each feature vs the training data | PSI (plus a KS test for reference) |
| Prediction drift | predicted effect of each discount | PSI |
| Behaviour drift | purchase rate of customers who got no discount | two-proportion z-test |

## Thresholds (rules of thumb)
- PSI below 0.10 ok, 0.10 to 0.25 warning, 0.25 or more alert
- Purchase rate: warning at a change of 2 points, alert at 5 points, and it must also be statistically clear (p below 0.01)
- At least 100 customers per check, and at least 50 customers without a discount for the purchase-rate check

Severity rests on PSI and effect size, not p-values, because with large
batches even tiny differences are statistically significant.

## Usage
- Command line: `python backend\src\drift.py new_customers.csv` (exit code 0 ok, 1 warning, 2 alert)
- API: `POST /drift` with customers (features, optionally `discount` and `purchased`)
- Demo: `python backend\src\drift_demo.py` writes `docs/drift_demo.csv` and `docs/drift_demo.png`

## Limits
- Checks each variable on its own, so it can miss changes in combinations of variables
- Predictions can lag behind input changes, so inputs and purchase rate are checked too
- The purchase-rate check assumes the business discounts the same kinds of customers as before; a policy change can move it without a real change in behaviour
- It cannot tell whether the effect of a discount has changed. That needs a randomized holdout
- PSI thresholds are conventions, not statistical guarantees