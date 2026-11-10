# EconoCausal: Dynamic Pricing via Double Machine Learning

A causal AI system that estimates, for each customer, how much a $10 or $20 discount changes their chance of buying, and then spends a fixed marketing budget only where discounts actually work.

Standard machine learning predicts who is likely to buy. That is the wrong question for a discount campaign: a customer with a 90% chance of buying may buy anyway, and the discount is wasted. EconoCausal answers a different question: *would this customer buy if, and only if, we gave them the discount?*

## Headline results

All results use synthetic data with known true effects (10,000 customers). Evaluation numbers come from customers the model never trained on, unless stated.

| Question | Result |
|---|---|
| Does confounding fool the naive comparison? | Yes. A $20 discount looks harmful (-0.097), but after causal adjustment the effect is +0.039, close to the true +0.042 |
| Can matching recover the effect? | 4,197 of 4,244 discounted customers matched; naive -0.084 became +0.047 against a true +0.049 |
| Can Double ML tell who responds? | Correlation with the true individual effects is 0.82 ($10) and 0.71 ($20). The top predicted-uplift decile truly gains +0.147; the bottom loses -0.078 |
| Does it beat a standard prediction model? | Qini coefficient 0.72 (causal forest) against 0.08 (standard purchase model) |
| Is the estimate robust? | Placebo, random-confounder and subsample refutation tests pass. Dropping a key confounder flips the estimate (-0.061), which shows the limit of such tests |
| Does budgeted targeting pay? | With a $1,500 budget on held-out customers: $1,667 of extra revenue against $206 for random discounts. Across budgets from $500 to $20,000 (per 10,000 customers) it earns 4.4 to 8.1 times more; the fast greedy method stays within about 1% of the exact optimum |

## Quick start (Windows PowerShell)

You need Git, Python 3.10 or newer (built with 3.13) and a current Node.js (built with Node 24).

```powershell
git clone https://github.com/anushreehd2003-cmyk/econocausal-dynamic-pricing.git
cd econocausal-dynamic-pricing
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
python backend\src\run_pipeline.py
```

The pipeline generates the data, trains and evaluates the models, exports the dashboard files and saves the model for the API. It takes roughly 10 minutes. Add `--all` to also run the analyses behind the documents in `docs/`, `--check` to see what is missing, or `--start-at SCRIPT.py` to resume after a failure. On macOS or Linux, activate with `source venv/bin/activate` and use `/` in paths.

Run the tests:

```powershell
python -m pytest backend\tests -v
cd frontend
npm install
npm test
```

Start the API (terminal 1, from the project root) and the dashboard (terminal 2):

```powershell
python -m uvicorn api:app --app-dir backend\src --port 8000
cd frontend
npm run dev
```

Open http://localhost:5173 for the dashboard and http://localhost:8000/docs for the interactive API documentation. The dashboard's demo data is committed, so it also runs without the pipeline; the API and the data health check need the saved model, which the pipeline creates.

## What it contains

| Module | Code |
|---|---|
| Causal inference engine (DoWhy, EconML) | `causal_graph.py`, `dml_engine.py`, `refutation.py` |
| Propensity score matching | `propensity.py`, `matching.py` |
| Optimization solver (SciPy) | `optimizer.py`, `efficiency.py` |
| Dashboard (React, Plotly) | `frontend/` |
| REST API (FastAPI) | `api.py`, `model_store.py`, `train_final_model.py` |
| Data drift detector | `drift.py`, `drift_demo.py` |

All Python code is in `backend/src`. See `docs/architecture.md` for how the pieces fit together.

### Dashboard

1. **Upload data:** load a campaign CSV, check its columns, and run a **data health check** against the API
2. **Budget:** set the budget and the allowed discount levels, with a live plan preview
3. **Uplift:** real Qini curves and filters over 10,000 predicted effects
4. **Allocation:** the prescription for your budget: who gets which discount, with search, filters, sorting, CSV export and a plain-language summary

![Upload](docs/screenshots/upload.png)
![Budget](docs/screenshots/budget.png)
![Uplift](docs/screenshots/uplift.png)
![Allocation](docs/screenshots/allocation.png)

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model status |
| `POST /predict` | Predicted effect of a $10 and a $20 discount per customer, with 90% intervals |
| `POST /allocate` | Discount prescription for a budget (`greedy` by default, `exact` is slower) |
| `POST /drift` | Compare new customers with the training data |

## Project structure

```
backend/src       data generator, causal models, optimizer, API, drift detector, pipeline runner
backend/tests     Python unit tests
backend/notebooks walkthrough notebook
data              generated datasets (only small samples are committed)
frontend          React dashboard (demo data in public/demo, JavaScript tests in tests/)
docs              analyses, architecture, screenshots
models            saved model for the API (generated, not committed)
```

## Documentation

- Design: `docs/architecture.md`, `docs/causal_assumptions.md`, `docs/data_dictionary.md`
- Analyses: `docs/propensity_analysis.md`, `docs/matching_analysis.md`, `docs/dml_analysis.md`, `docs/refutation_analysis.md`, `docs/model_comparison.md`, `docs/optimizer_analysis.md`, `docs/efficiency_analysis.md`
- Components: `docs/api.md`, `docs/drift_detection.md`
- Weekly summaries: `docs/week1_summary.md`, `docs/week2_summary.md`, `docs/week3_summary.md`

## Limitations

- The data is synthetic. True individual effects are known only for that reason, which is how the models and the Qini curves are evaluated. Real data would need a randomized holdout group.
- The causal estimates assume that every variable affecting both the discount and the purchase is in the model. Refutation tests support this assumption but cannot prove it.
- Revenue figures are gross: they ignore margin given to customers who would have bought anyway, so they are not profit.
- Predictions shown in the dashboard are in-sample for about 70% of the customers. The held-out figures in this README are the fair test.
- The live dashboard allocation uses a fast greedy method; the exact optimizer is available through the API.
- The API has no authentication and is meant to run locally.
- Drift checks look at each variable separately and can miss changes in combinations of variables.

## Progress

- [x] Week 1: causal graph (DoWhy) and dashboard scaffolding
- [x] Week 2: propensity scores, matching, Double ML and uplift curves
- [x] Week 3: SciPy budget optimizer and allocation view
- [ ] Week 4: REST API and drift detector built; final polish and documentation in progress