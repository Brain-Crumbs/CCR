# Market evaluation (template only)

Run: <exact identity> — <completed/failed and reason>
Population: <synthetic/play_money/financial_market>; partition: <validation/test>

## Coverage first

<Sample, realized, censored, abstained, stale, missing and scored counts>
<Per-target/horizon/source/instrument populations; retain every requested ID>

Exact baseline: <name, model identity and payload hash>

## Limitations

- <Pretraining cutoff or unknown; contamination limitations>
- <Historical survivorship and source coverage>
- <Synthetic generator recovery is not financial-market evidence>
- Forecast quality only; no P&L, tradable edge or live-validation claim.

## Scores and uncertainty

<Quantile levels; pinball, explicit median MAE, outer interval coverage/width>
<Direction Brier only for explicit head; no resolution Brier or sparse-quantile CRPS>
<N and time-block interval or explicit insufficient-time-blocks status per score>

## Paired baseline difference

<Jointly scored forecast IDs, N, pinball improvement and time-block uncertainty>

## Full reproducible report

<JSON from make_report including hashes, slices, status and retained populations>
