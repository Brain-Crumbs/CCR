"""Coverage-first JSON/Markdown reports; sealed test artifacts are never objectives."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from cognitive_runtime.data.ledger import digest
from .population import bind_population
from .metrics import score
from .bootstrap import time_block_interval


METRICS = ('pinball', 'median_mae', 'interval_coverage', 'interval_width', 'direction_brier')


def _summary(rows, bootstrap):
    usable = [r for r in rows if r.outcome.label is not None and not r.result.abstention_reason]
    scores = [(r, score(r.result.forecast, r.outcome.label.value)) for r in usable]
    counts = {'samples': len(rows), 'realized': sum(r.outcome.label is not None for r in rows),
              'censored': sum(r.outcome.label is None for r in rows),
              'abstained': sum(bool(r.result.abstention_reason) for r in rows),
              'stale': sum(r.stale for r in rows), 'missing': sum(r.missing for r in rows), 'scored': len(usable)}
    metrics = {}
    for metric in METRICS:
        values = [(r.decision_at, s[metric]) for r, s in scores if metric in s]
        metrics[metric] = {'n': len(values), 'value': sum(v for _, v in values)/len(values) if values else None,
                           'uncertainty': time_block_interval(values, **bootstrap)}
    grids = sorted({tuple(q.level for q in r.result.forecast.quantiles) for r in usable})
    if len(grids) > 1:
        raise ValueError('cannot aggregate different quantile grids')
    return {'counts': counts, 'coverage': len(usable)/len(rows) if rows else None,
            'censor_reasons': dict(Counter(r.outcome.censor_reason for r in rows if r.outcome.censor_reason)),
            'abstention_reasons': dict(Counter(r.result.abstention_reason for r in rows if r.result.abstention_reason)),
            'quantile_levels': grids[0] if grids else (), 'metrics': metrics}


def make_report(corpus, plan, predictions, *, baseline, population, baseline_name, run_id,
                partition='test', status='completed', failure=None, pretraining_cutoff='unknown',
                caveats=(), slices=None, bootstrap=None):
    """Caller supplies predeclared regime/liquidity/missingness slice membership.

    Group-disjoint guarantees come only from SplitPlan.group, not from a slice
    label. Report every ID even for failure; supply explicit abstained results.
    """
    if population not in ('synthetic', 'play_money', 'financial_market'):
        raise ValueError('explicit evaluation population required')
    if status not in ('completed', 'failed') or (status == 'failed') != bool(failure):
        raise ValueError('failed run requires failure reason')
    if not baseline_name or not run_id or not pretraining_cutoff:
        raise ValueError('exact baseline/run/cutoff metadata required')
    bootstrap = dict(bootstrap or {})
    rows = bind_population(corpus, plan, predictions, partition=partition)
    refs = bind_population(corpus, plan, baseline, partition=partition)
    if any(r.synthetic != (population == 'synthetic') for r in rows):
        raise ValueError('synthetic and real evaluation populations must remain separate')
    # Protocols distinguish real play-money vs financial prices explicitly;
    # provenance.synthetic alone cannot establish market type.
    all_ids = {r.result.sample_id for r in rows}
    slices = slices or {}
    if not set(slices) <= {'regime', 'liquidity', 'missingness'}:
        raise ValueError('unsupported slice dimension')
    normalized = {}
    for dimension in ('regime', 'liquidity', 'missingness'):
        groups = slices.get(dimension, {'unknown': tuple(sorted(all_ids))})
        seen = []
        for name, ids in groups.items():
            if not name or len(set(ids)) != len(ids) or not set(ids) <= all_ids:
                raise ValueError('invalid slice membership')
            seen.extend(ids)
        if set(seen) != all_ids or len(seen) != len(all_ids):
            raise ValueError('slices must account for each population ID exactly once')
        normalized[dimension] = {name: _summary([r for r in rows if r.result.sample_id in ids], bootstrap)
                                 for name, ids in groups.items()}
    for dimension, attr in [('issuer', 'issuer_ids'), ('event_family', 'event_families')]:
        names = sorted({v for r in rows for v in getattr(r, attr)})
        normalized[dimension] = {name: _summary([r for r in rows if name in getattr(r, attr)], bootstrap) for name in names}
        normalized[dimension]['unknown'] = _summary([r for r in rows if not getattr(r, attr)], bootstrap)
    paired = []
    paired_ids = []
    for r, b in zip(rows, refs):
        if r.result.forecast_id != b.result.forecast_id or r.label_sha256 != b.label_sha256:
            raise ValueError('baseline pair mismatch')
        if r.outcome.label is not None and not r.result.abstention_reason and not b.result.abstention_reason:
            if tuple(q.level for q in r.result.forecast.quantiles) != tuple(q.level for q in b.result.forecast.quantiles):
                raise ValueError('paired quantile grids differ')
            paired.append((r.decision_at, score(b.result.forecast, b.outcome.label.value)['pinball'] -
                           score(r.result.forecast, r.outcome.label.value)['pinball']))
            paired_ids.append(r.result.forecast_id)
    summary = _summary(rows, bootstrap)
    summary['baseline'] = {'name': baseline_name, 'model_ids': sorted({r.result.forecast.model_id for r in refs if r.result.forecast}),
                           **_summary(refs, bootstrap)}
    summary['paired_pinball_improvement'] = {'n': len(paired), 'forecast_ids': paired_ids,
        'value': sum(v for _, v in paired)/len(paired) if paired else None,
        'uncertainty': time_block_interval(paired, **bootstrap)}
    buckets = {}
    for r in rows:
        buckets.setdefault((r.source, r.instrument), []).append(r)
    limitations = ['No tradable-edge or live-validation claim; forecast quality only; P&L not computed.',
        'Pretraining cutoff: ' + pretraining_cutoff + '; model identity is not absence of contamination.',
        'Historical survivorship, source coverage and missing instruments may limit generalization.',
        'Time-block interval depends on chosen block length and is unreliable for small populations.',
        'CRPS unavailable: sparse quantiles do not define a full distribution.'] + list(caveats)
    if population == 'synthetic':
        limitations.insert(0, 'synthetic_only: generator recovery/null controls are not financial-market evidence.')
    report = {'format': 'market-evaluation-v1', 'run_id': run_id, 'status': status, 'failure': failure,
        'population': population, 'partition': partition, 'sealed': partition == 'test', 'edge_promotable': False,
        'pnl': None, 'summary': summary, 'caveats': limitations,
        'identities': {'corpus_sha256': corpus.manifest_sha256, 'split_sha256': plan.hash,
                       'label_spec_sha256': digest(corpus.manifest['spec']),
                       'prediction_sha256': digest([asdict(r.result) for r in rows]),
                       'baseline_sha256': digest([asdict(r.result) for r in refs]),
                       'labels': {r.result.sample_id: r.label_sha256 for r in rows}},
        'target': corpus.manifest['spec'], 'group_disjoint_policy': plan.group,
        'by_source_instrument': [{'source': source, 'instrument': instrument, **_summary(group, bootstrap)}
                                for (source, instrument), group in sorted(buckets.items())],
        'slices': normalized,
        'population_rows': [{'sample_id': r.result.sample_id, 'forecast_id': r.result.forecast_id,
                            'decision_at': r.decision_at, 'censored': r.outcome.censor_reason,
                            'abstained': r.result.abstention_reason} for r in rows]}
    return report


def validation_objective(report, *, metric, predeclared_metric):
    if report['partition'] != 'validation' or report['sealed'] or report['status'] != 'completed':
        raise ValueError('only completed validation reports may score Factory search')
    if metric != predeclared_metric or metric not in ('pinball', 'median_mae', 'direction_brier'):
        raise ValueError('metric must match predeclared validation loss')
    value = report['summary']['metrics'][metric]['value']
    if value is None:
        raise ValueError('validation metric unavailable')
    return value


def markdown(report):
    summary = report['summary']
    lines = ['# Market evaluation', '', f"Run: {report['run_id']} — {report['status']}; population: {report['population']}; partition: {report['partition']}", '',
             '## Coverage first', '', json.dumps(summary['counts'], sort_keys=True), '',
             f"Exact baseline: {summary['baseline']['name']}; models: {summary['baseline']['model_ids']}", '',
             '## Limitations', ''] + ['- ' + c for c in report['caveats']]
    lines += ['', '## Scores and uncertainty', '', '```json', json.dumps(summary['metrics'], indent=2), '```', '',
              '## Paired baseline difference', '', '```json', json.dumps(summary['paired_pinball_improvement'], indent=2), '```', '',
              '## Full reproducible report', '', '```json', json.dumps(report, indent=2), '```', '']
    return '\n'.join(lines)


def seal_report(report, path):
    """Exclusive creation prevents replacing a locked test report at this path.

    This is an accidental-overwrite guard, not a security boundary against an
    operator who can delete files or run arbitrary Python. Search uses the
    validation_objective gate, never a test report.
    """
    payload = {'sha256': digest(report), 'report': report}
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
    return payload['sha256']
