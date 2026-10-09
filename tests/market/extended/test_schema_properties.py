"""Opt-in seeded property corpus; stdlib only, no implicit package installation."""
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
import json
from pathlib import Path
import random
import statistics
import unittest
from cognitive_runtime.adapters.finance.schemas import Target, parse_record
from cognitive_runtime.adapters.finance.schemas.base import timestamp
from cognitive_runtime.adapters.finance.schemas.records import Quantile
from cognitive_runtime.adapters.finance.schemas.semantics import price_change, support
from cognitive_runtime.adapters.finance.schemas.migrations import migrate_legacy_0_9
from cognitive_runtime.adapters.finance.testing.examples import examples, T
from cognitive_runtime.adapters.finance.testing.generate import generate


class SchemaProperties(unittest.TestCase):
    def test_seeded_decimal_and_distribution_boundaries(self):
        rng = random.Random(286)
        forecast = examples()['ForecastDistribution']
        for _ in range(1000):
            anchor, end = (Decimal(rng.randrange(1000001)) / 1000000 for _ in range(2))
            a, b = format(anchor, 'f'), format(end, 'f')
            self.assertEqual(price_change(Target.CONTRACT_CHANGE, a, b), end - anchor)
            lo, hi = support(Target.CONTRACT_CHANGE, a)
            with localcontext() as ctx:
                ctx.prec = 2
                self.assertEqual(support(Target.CONTRACT_CHANGE, a), (lo, hi))
            replace(forecast, anchor_price=a, support_lower=str(lo), support_upper=str(hi),
                    quantiles=(Quantile(level='0.1', value=str(lo)), Quantile(level='0.9', value=str(hi))))
            with self.assertRaises(ValueError):
                replace(forecast, anchor_price=a, support_lower=str(lo), support_upper=str(hi),
                        quantiles=(Quantile(level='0.5', value=str(hi + Decimal('0.000001'))),))

    def test_seeded_timestamp_replay(self):
        rng = random.Random(287)
        for _ in range(1000):
            dt = timestamp(T) + timedelta(seconds=rng.randrange(-10000000, 10000000), microseconds=rng.randrange(1000000))
            wire = dt.isoformat().replace('+00:00', 'Z')
            self.assertEqual(timestamp(wire), dt)
            with self.assertRaises(ValueError):
                timestamp(wire[:-1])

    def test_supported_prior_version_corpus_and_new_semantics_rejection(self):
        path = Path(__file__).resolve().parents[2] / 'fixtures/market/prior-0.9.json'
        raw = json.loads(path.read_text())
        before = path.read_bytes()
        current = migrate_legacy_0_9(raw)
        self.assertEqual(parse_record(current.to_dict()), current)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            migrate_legacy_0_9({**raw, 'schema_version': '2.0.0'})

    def test_optional_large_signal_oracle_and_null_expectations(self):
        # Verify the actual seeded process, not predictive gain or an external model.
        signal = generate(steps=2048, lag=2)
        null = generate(steps=2048, lag=2, zero_signal=True)
        correlations = []
        for payload in (signal, null):
            oracle = payload['training_only_oracle']
            latent = [r['latent'] for r in oracle[1:]]
            changes = [b['equity_log'] - a['equity_log'] for a, b in zip(oracle, oracle[1:])]
            correlations.append(statistics.correlation(latent, changes))
            for record in payload['records']:
                parse_record(record)
        self.assertGreater(correlations[0], 0.9)
        self.assertLess(abs(correlations[1]), 0.1)
        self.assertEqual([x['text_reading'] for x in signal['training_only_oracle']],
                         [x['text_reading'] for x in null['training_only_oracle']])


if __name__ == '__main__':
    unittest.main()
