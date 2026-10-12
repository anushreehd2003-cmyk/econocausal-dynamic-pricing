# Causal Assumptions

- **Treatment:** discount (0, 10, 20 dollars)
- **Outcome:** purchased (1 = bought during the campaign)
- **Confounders:** age, income, loyalty_member, past_purchases,
  days_since_last_purchase. They influence both who received a discount
  and who buys.
- **Unconfoundedness:** we assume no hidden variable affects both discount
  and purchase beyond these. DoWhy refutation tests in the Mid Review will
  stress-test this.
- **Excluded:** avg_basket_value (affects revenue only);
  ground_truth_effects.csv (validation only, never a model input).
- **Why the naive estimate fails:** discounts went mostly to inactive
  customers, who buy less anyway, so raw comparisons make discounts look
  harmful.