# Week 1 Summary

## Delivered so far
- Synthetic retail dataset with confounded discount assignment (10,000 customers)
- Ground-truth individual effects, used only for validation
- Causal DAG in DoWhy (treatment: discount, outcome: purchased, 5 confounders)
- Backdoor identification and a linear adjusted estimate
- React dashboard: Upload, Budget and Uplift tabs
- Unit tests for the data generator and the DAG
- Walkthrough notebook: backend/notebooks/01_causal_walkthrough.ipynb

## Key result
The naive comparison shows discounts lowering purchases. After backdoor
adjustment the average effect is close to the true value, confirming that
the DAG captures the confounding.

## Next (Week 2)
Propensity score estimation and matching, Double ML training with EconML,
real Qini and uplift curves in the dashboard.