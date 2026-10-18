"""Explicit tiny synthetic smoke. No providers, downloads or neural models."""
import argparse
import json
from pathlib import Path
import yaml
from cognitive_runtime.adapters.finance.testing.factory import prepare_synthetic
from cognitive_runtime.adapters.finance.factory import market_spec, final_test
from cognitive_runtime.adapters.finance.models.baselines import BaselineConfig
from cognitive_runtime.training.model_factory.runner import run_trial
from cognitive_runtime.training.model_factory.artifacts import atomic_write_json, _jsonable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/market/factory-smoke.yaml'))
    parser.add_argument('--final-test', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.config.read_text())
    if set(cfg) != {'seed', 'baseline', 'max_training_seconds'} or type(cfg['seed']) is not int:
        raise ValueError('invalid smoke configuration')
    corpus, plan = prepare_synthetic(args.output / 'private', cfg['seed'])
    spec = market_spec(corpus, plan, seed=cfg['seed'], baseline=BaselineConfig(**cfg['baseline']),
                       seconds=cfg['max_training_seconds'])
    atomic_write_json(args.output / 'resolved-spec.json', _jsonable(spec.to_dict()))
    result = run_trial(spec, root=args.output / 'runs', corpus_root=corpus.root.parent)
    if args.final_test:
        final_test(result, corpus_root=corpus.root.parent, root=args.output / 'runs' / spec.organism)
    print(json.dumps({'run_id': result.run_id, 'status': result.state,
                      'manifest': str(result.directory / 'run_manifest.json')}))


if __name__ == '__main__':
    main()
