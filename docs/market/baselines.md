# Offline distributional baselines (#293)

These NumPy models predict price change from the existing #291 snapshots and
#292 hashing/TF-IDF features. They are supervised reference baselines, not JEPA.
No network, keys, local weights or optional neural packages are required.
Provider capability remains **synthetic_only**; there is no live validation.

```python
from cognitive_runtime.adapters.finance.models.baselines import BaselineConfig, BaselineModel
from cognitive_runtime.adapters.finance.models.fitting import fit_baseline

# corpus and plan are the persisted #291 Corpus and validated SplitPlan.
config = BaselineConfig(kind='multimodal', text_mode='tfidf', text_dimensions=16)
model = fit_baseline(corpus, plan, listing_id='contract:demo', config=config)
results = model.forecast([corpus.feature(key) for key in plan.test])
model.save('baseline.json')
resumed = BaselineModel.load('baseline.json', corpus_sha256=corpus.manifest_sha256,
                            split_sha256=plan.hash, config=config)
assert resumed.hash == model.hash
```

The configuration inventory is `config/market/models/baselines.json`. Each kind
uses the same corpus, split, label spec and forecast pairing IDs:

| Kind | Definition |
| --- | --- |
| zero | Constant zero-change quantiles, explicitly labeled baseline. |
| unconditional | Empirical train-label quantiles, represented by mean plus residuals. |
| volatility | Zero-centered train residual quantiles standardized by context RMS; scale floor is training-label RMS (at least 1e-8). |
| numeric | Ridge regression on last change, context change RMS, relative anchor age and missing-history mask. |
| text | Ridge regression on mean selected hash/TF-IDF features plus missing-text mask. |
| multimodal | Ridge regression on concatenated numeric/text vectors. |

All fitted heads use an unpenalized intercept, train-only scaling and empirical
train-residual quantiles. Ridge applies to the three regression kinds only.
Quantiles are monotone and clipped to exact Decimal target support: equity
simple returns ≥ -1; contract changes in [-anchor, 1-anchor]. Clipping creates
boundary mass and can reduce interval width. These are **uncalibrated** forecasts:
no direction probability, calibration, Gaussian assumption or coverage claim.
Calibration, metrics and formal null/lag/shuffle controls belong to #294.

Inputs are bounded to 512 train samples, 64 text dimensions and 19 quantiles.
This is a small CPU reference implementation, not a large corpus training engine.
`predict` implements the neutral `PredictiveModel`/`InferenceInput` interface;
`forecast` wraps it in financial schemas with pairing, snapshot, label-spec and
split identities. Factory lifecycle registration/search/reporting is #295. The
Crafter backend and its serialization are unchanged.

Fit reads only validated train outcomes. Inference reads selected context only;
it never opens outcome or terminal stores. A future known expiry, stale anchor or
quarantined latest observation causes abstention. If no valid anchor exists,
`ForecastResult.forecast` is null and the reason is explicit: schema-required
anchor metadata is never invented. Missing text is a masked zero vector. Label
censoring is independent from prediction abstention and must be reported by the
later evaluator without dropping population IDs.

Checkpoints retain fitted coefficients, scaler, train residuals, text vocabulary
and IDF, training membership/target hash, effective config and corpus/split hashes.
Loading rejects changes to those requested identities and checks content integrity.
This closed-form fit has no optimizer cursor: resume restores the finished fit
exactly, without refitting on validation/test. Checkpoints are trusted local
artifacts, not a security boundary against a writer who can recompute hashes.

Run lightweight acceptance with:

```sh
python -m pytest tests/market/test_distributional_baselines.py tests/market/test_text_features.py -q
python -m pytest tests/market/extended/test_baseline_recovery.py --run-market=extended -q
```

The optional tiny recovery test uses project-authored paths and fixed settings.
Any supervised success means **generator recovery only**, never market edge.
Default CI remains offline and torch-free, with the existing aggregate 120-second
budget. No real provider payload or model download is needed.
