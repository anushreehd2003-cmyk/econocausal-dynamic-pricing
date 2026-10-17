# Propensity Score Analysis (Week 2, Day 8)

- **Treatment (binary):** any discount vs none
- **Model:** logistic regression on the five confounders, 5-fold cross-fitted
- **AUC:** <your value>
- **Largest imbalance before weighting:** <covariate> (SMD <value>)
- **Largest imbalance after IPW:** <covariate> (SMD <value>)
- **Effective sample size after IPW:** <value> of 10000

![Overlap](propensity_overlap.png)
![Love plot](love_plot_ipw.png)

## Notes
- Cross-fitting avoids overfitting the propensity scores to the same rows.
- Balance is checked on observed covariates only. Unobserved confounding
  is tested later with DoWhy refutation tests.
- The multi-level treatment ($10 vs $20) is handled by Double ML.