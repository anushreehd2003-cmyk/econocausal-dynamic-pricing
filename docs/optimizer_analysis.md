# Budget Optimizer (Week 3, Day 15)

- **Problem:** choose $0, $10 or $20 per customer to maximize predicted incremental revenue, with total discount face value at most the budget
- **Method:** exact integer program (scipy.optimize.milp), one discount per customer
- **Value of a discount:** predicted change in purchase probability x average basket value (gross incremental revenue)
- **Baselines:** greedy by value per dollar; blanket $10 to random customers; oracle (exact solve using the true effects)

## Held-out evaluation (3,000 customers, budget $1,500 = $5,000 scaled)

| Strategy | Targeted | Spend | Predicted extra revenue | True extra revenue | True extra purchases | Share with positive true effect |
|---|---|---|---|---|---|---|
| Optimizer (exact) | 138 | $1,500 | $1,553 | $1,667 | 15.95 | 98.6% |
| Greedy | 141 | $1,500 | $1,545 | $1,687 | 16.15 | 98.6% |
| Blanket $10 | 150 | $1,500 | $90 | $206 | 3.63 | 62.7% |
| Oracle (true effects) | 149 | $1,500 | $1,272 | $2,128 | 22.24 | 100% |

## All 10,000 customers, budget $5,000
- $10 discounts: 416, $20 discounts: 42, spend $5,000
- Predicted extra revenue $5,161 and predicted extra purchases 47.9. Predictions are in-sample for the 70% of customers the model trained on.

## Findings
- Optimized targeting earns about 8 times the extra revenue of blanket targeting at the same budget and reaches about 78% of the oracle ceiling.
- Greedy is within about 1% of the exact optimum on true revenue, so it is suitable for live use in the dashboard.
- Predicted revenue is about 7% below true revenue for the optimizer's picks, so there is no sign of overconfidence. The model underestimates the very best customers (the oracle's picks) by about 40%.
- Almost all of the budget goes to $10 discounts because the effect of $20 is only about 1.4 times that of $10.

## Limits
- Revenue is gross. It ignores margin given away to customers who would have bought anyway, so absolute ROI or profit cannot be read from it. Compare strategies at the same budget.
- The oracle comparison needs true effects, which only synthetic data provides.
- The exact solve took several seconds for 10,000 customers, which is too slow for an interactive slider.