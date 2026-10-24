# Week 2 Summary

## Delivered
- Propensity scores and balance diagnostics (backend/src/propensity.py)
- Propensity score matching with caliper (backend/src/matching.py)
- Double ML with EconML, causal forest and linear variants (backend/src/dml_engine.py)
- Causal audit with DoWhy and Double ML refutation tests (backend/src/refutation.py)
- Model comparison and Qini curves (backend/src/model_comparison.py)
- Dashboard Uplift tab with real Qini curves and filters over 10,000 predicted effects
- 18 unit tests

## Key results
| Step | Result |
|---|---|
| Propensity | AUC 0.718; largest imbalance 0.751 -> 0.022 after weighting |
| Matching | 4197/4244 matched; ATT +0.047 (CI +0.028 to +0.066); true +0.049 |
| Double ML (held-out) | correlation with true effects 0.82 ($10), 0.71 ($20) |
| Qini coefficient | causal forest 0.721, linear DML 0.715, standard model 0.078 |
| Refutation | placebo +0.005 (p 0.315); random cause +0.036; subset +0.038; omitting the key confounder gives -0.061 in Double ML |

## Limitations
- Data is synthetic; Qini curves use the known true effects, which real data would not provide
- Unconfoundedness is assumed; refutation tests support but cannot prove it
- Predictions for the lowest deciles are shrunk towards zero
- econml warns that linear DML inference is invalid, so intervals come from the forest only

## Next (Week 3)
SciPy budget optimizer and the allocation matrix view in the dashboard.