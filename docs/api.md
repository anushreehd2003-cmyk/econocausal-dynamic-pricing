# REST API (Week 4, Day 19)

Built with FastAPI. The model is trained on all 10,000 customers by
`backend/src/train_final_model.py` and saved to `models/dml_forest.joblib`
(not committed). Start the server with
`python -m uvicorn api:app --app-dir backend\src --port 8000`.

| Endpoint | Purpose |
|---|---|
| GET /health | Model status, training date, library versions (503 if the model file is missing) |
| POST /predict | Predicted change in purchase probability for a $10 and a $20 discount, with 90% intervals |
| POST /allocate | Allocates discounts within a budget; `method` is `greedy` (default) or `exact` |

Inputs per customer: age, income, loyalty_member (0 or 1), past_purchases,
days_since_last_purchase, plus avg_basket_value for /allocate. At most 10,000
customers per request. Invalid input returns 422.

## Notes
- Customers outside the training feature ranges are flagged in `warnings`.
- The exact solver is limited to 20 seconds and may return the best solution found.
- The model file is a pickle: only load files you created yourself. The API reports a warning if library versions differ from training.
- CORS allows the Vite dev server (localhost:5173) so the dashboard can call the API.
- There is no authentication, so run it locally only.

## Dashboard integration
The Upload tab has a Data health check. It sends the uploaded file to `POST /drift`
(features plus `discount` and `purchased`, at most 10,000 valid rows, at least 100)
and shows the status, plain-language messages and a table of PSI values.
Rows with missing or invalid values are left out and counted. The dashboard reads
the API address from `VITE_API_URL` (default `http://localhost:8000`).
The Allocation tab does not call the API: it uses the precomputed predictions
and the in-browser greedy allocator so that budget changes respond instantly.