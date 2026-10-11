# Data Dictionary (retail_campaign.csv)

| Column | Role | Description |
|---|---|---|
| customer_id | ID | Unique customer |
| age | Confounder | Age in years |
| income | Confounder | Annual income |
| loyalty_member | Confounder | 1 if loyalty member |
| past_purchases | Confounder | Previous purchase count |
| days_since_last_purchase | Confounder | Recency |
| avg_basket_value | Confounder | Typical order value |
| discount | Treatment | 0, 10 or 20 (dollars) |
| purchased | Outcome | 1 if bought during campaign |
| revenue | Outcome | Order revenue (0 if no purchase) |

`ground_truth_effects.csv` holds the true uplift per customer. It is used
only to validate the model and must never be used as a model input.