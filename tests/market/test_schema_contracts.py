"""Offline schema acceptance: handwritten mutations, real parsers and artifacts."""
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cognitive_runtime.adapters.finance.schemas import *
from cognitive_runtime.adapters.finance.schemas.base import decimal, timestamp
from cognitive_runtime.adapters.finance.schemas.records import Quantile, DirectionProbability
from cognitive_runtime.adapters.finance.schemas.semantics import price_change
from cognitive_runtime.adapters.finance.labels import LabelSpec, RealizedLabel, ContractTerminalMetadata, FutureTarget
from cognitive_runtime.adapters.finance.inference import validate_context, to_inference_input
from cognitive_runtime.adapters.finance.testing.examples import examples, T, END
from cognitive_runtime.adapters.finance.testing.generate import generate, edge_cases, write_fixtures, FixtureManifest, digest
from cognitive_runtime.adapters.finance.testing.document import render, schema_for

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'tests/fixtures/market'


class SchemaContracts(unittest.TestCase):
    def setUp(self):
        self.e = examples()

    def test_all_families_round_trip_and_deep_unknown_fields(self):
        for name, record in self.e.items():
            with self.subTest(name=name):
                cls = type(record)
                self.assertEqual(cls.from_json(record.canonical_json()), record)
                self.assertEqual(cls.from_json(record.canonical_json()).hash, record.hash)
                raw = record.to_dict()
                raw['future_terminal_payout'] = '1'
                with self.assertRaises(TypeError):
                    cls(**raw)
                raw = record.to_dict()
                raw['provenance']['terminal'] = '1'
                with self.assertRaises(TypeError):
                    cls(**raw)
                with self.assertRaises(ValueError):
                    replace(record, schema_version='9.0.0')

    def test_generated_schemas_and_docs_match_executable_source(self):
        for path, expected in render(ROOT).items():
            self.assertEqual(path.read_text(), expected, str(path))
        import jsonschema
        for record in self.e.values():
            schema = schema_for(type(record))
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.validate(record.to_dict(), schema)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({**record.to_dict(), 'terminal': True}, schema)

    def test_decimal_semantics_and_ambient_precision(self):
        with localcontext() as ctx:
            ctx.prec = 2
            self.assertEqual(price_change(Target.CONTRACT_CHANGE, '0.42', '0.47'), Decimal('0.05'))
            self.assertEqual(price_change(Target.EQUITY_RETURN, '100', '105'), Decimal('0.05'))
            self.assertEqual(price_change(Target.CONTRACT_CHANGE, '0.123456', '0.223456'), Decimal('0.1'))
        self.assertEqual(price_change(Target.CONTRACT_CHANGE, '0.42', '0.47') * 100, 5)
        for bad in (0.42, 'NaN', 'Infinity', '1e-3', '+1', '01', True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decimal(bad)
        with self.assertRaises(ValueError):
            price_change(Target.EQUITY_RETURN, '0', '1')
        with self.assertRaises(ValueError):
            price_change(Target.CONTRACT_CHANGE, '0.5', '1.1')

    def test_timestamp_and_precision_fail_closed(self):
        for bad in ('2025-01-01', '2025-01-01T00:00:00', '2025-01-01T00:00:00+01:00', '2025-01-01T00:00:00.1234567Z'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                timestamp(bad)
        with self.assertRaises(ValueError):
            replace(self.e['Quote'], event_at='2025-01-06T14:30:00.1Z')
        with self.assertRaises(ValueError):
            replace(self.e['Quote'], ingested_at='2025-01-05T14:30:00Z')

    def test_forecast_nonmonotone_and_out_of_support_fail(self):
        forecast = self.e['ForecastDistribution']
        for qs in ((Quantile(level='0.9', value='0'), Quantile(level='0.1', value='0.1')),
                   (Quantile(level='0.1', value='0.1'), Quantile(level='0.9', value='0')),
                   (Quantile(level='0.5', value='0.59'),), (Quantile(level='0.5', value='-0.43'),)):
            with self.subTest(qs=qs), self.assertRaises(ValueError):
                replace(forecast, quantiles=qs)
        with self.assertRaises(ValueError):
            replace(forecast, support_upper='1')
        with self.assertRaises(ValueError):
            replace(forecast, unit=Unit.RETURN)
        with self.assertRaises(TypeError):
            ForecastDistribution(**{**forecast.to_dict(), 'resolution_probability': '0.8'})

    def test_equity_forecast_support_is_distinct(self):
        f = replace(self.e['ForecastDistribution'], target=Target.EQUITY_RETURN, unit=Unit.RETURN,
                    horizon=Horizon(clock=Clock.SESSIONS, count=1, calendar_id='synthetic', calendar_version='1'),
                    anchor_price='100', support_lower='-1', support_upper=None)
        with self.assertRaises(ValueError):
            replace(f, quantiles=(Quantile(level='0.5', value='-1.01'),))
        replace(f, quantiles=(Quantile(level='0.5', value='4'),))

    def test_direction_requires_separate_calibration(self):
        direction = DirectionProbability(positive_change_probability='0.6', method='separately_calibrated_head', calibrator_id='validation:v1')
        replace(self.e['ForecastDistribution'], direction=direction)
        with self.assertRaises(ValueError):
            replace(direction, method='inferred_from_three_quantiles')
        with self.assertRaises(ValueError):
            replace(direction, positive_change_probability='1.1')

    def test_explicit_abstention_has_no_outputs(self):
        f = self.e['ForecastDistribution']
        with self.assertRaises(ValueError):
            replace(f, quantiles=())
        with self.assertRaises(ValueError):
            replace(f, abstention_reason='stale')
        replace(f, quantiles=(), abstention_reason='stale')

    def test_horizon_clocks_and_origin_are_explicit(self):
        for raw in ({'clock': 'hour', 'count': 1, 'calendar_id': None, 'calendar_version': None},
                    {'clock': 'exchange_sessions', 'count': 1, 'calendar_id': None, 'calendar_version': None},
                    {'clock': 'utc_elapsed_seconds', 'count': 0, 'calendar_id': None, 'calendar_version': None}):
            with self.assertRaises(ValueError):
                Horizon(**raw)
        with self.assertRaises(ValueError):
            replace(self.e['LabelSpec'].horizon, origin='anchor_at')

    def test_label_currency_feed_mark_and_stale_anchor(self):
        label = self.e['RealizedLabel']
        for change in ({'endpoint_currency': 'EUR'}, {'endpoint_feed': 'other'}, {'endpoint_mark': Mark.BID},
                       {'anchor_at': '2025-01-06T14:00:00Z'}, {'value': '0.119047619'},
                       {'nominal_endpoint_at': '2025-01-06T15:15:00Z'}, {'actual_endpoint_at': '2025-01-06T15:32:00Z'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(label, **change)

    def test_split_neutral_ex_dividend_label(self):
        spec = replace(self.e['LabelSpec'], target=Target.EQUITY_RETURN, unit=Unit.RETURN,
                       horizon=Horizon(clock=Clock.TRADING_HOURS, count=1, calendar_id='synthetic', calendar_version='1'),
                       action_policy='split_neutral_ex_dividend')
        label = replace(self.e['RealizedLabel'], spec=spec, anchor_price='100', endpoint_price='52.5',
                        split_multiplier='2', action_record_ids=('split:demo',), value='0.05')
        self.assertEqual(label.value, '0.05')
        with self.assertRaises(ValueError):
            replace(label, action_record_ids=())
        self.assertEqual(self.e['PriceBar'].close, '100')

    def test_censored_labels_never_fill_with_settlement_or_zero(self):
        label = self.e['RealizedLabel']
        with self.assertRaises(ValueError):
            replace(label, censor_reason='early_closure')
        censored = replace(label, censor_reason='early_closure', endpoint_record_id=None, endpoint_price=None,
                           actual_endpoint_at=None, value=None, target_available_at=None)
        with self.assertRaises(ValueError):
            replace(censored, value='0')

    def test_market_snapshots_bind_real_neutral_data_contract(self):
        from cognitive_runtime.adapters.finance.contracts import snapshot_data_contract
        from cognitive_runtime.training.model_factory.task_contracts import TaskIdentity, TaskDataContract
        identity = TaskIdentity('market', 'price-change', 'synthetic-fixture')
        snapshot = self.e['ContextSnapshot']
        contract = snapshot_data_contract(identity, 'synthetic-286', train=(snapshot,))
        self.assertIsInstance(contract, TaskDataContract)
        self.assertEqual(contract.split_artifacts['train'][0]['sha256'], snapshot.hash)
        self.assertEqual(contract.feature_schema, snapshot.feature_schema)
        with self.assertRaises(ValueError):
            snapshot_data_contract(identity, 'synthetic-286', train=(snapshot,), test=(snapshot,))

    def test_context_and_neutral_inference_boundary(self):
        snapshot = self.e['ContextSnapshot']
        observations = [self.e['PriceBar'], self.e['NewsEvent']]
        inputs = to_inference_input(snapshot, observations, (100.0, 1.0))
        self.assertEqual(inputs.sample_ids, (snapshot.record_id,))
        self.assertEqual(inputs.feature_schema, snapshot.feature_schema)
        for name in ('ContractTerminalMetadata', 'FutureTarget', 'RealizedLabel', 'LabelSpec'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    parse_record(self.e[name].to_dict())
                with self.assertRaises(TypeError):
                    validate_context(snapshot, [self.e[name]])
        with self.assertRaises(TypeError):
            ContextSnapshot(**{**snapshot.to_dict(), 'terminal': self.e['ContractTerminalMetadata'].to_dict()})
        with self.assertRaises(TypeError):
            to_inference_input(snapshot, observations, ({'payout': 1},))

    def test_context_hash_and_clock_binding(self):
        snapshot = self.e['ContextSnapshot']
        bar, news = self.e['PriceBar'], self.e['NewsEvent']
        with self.assertRaises(ValueError):
            validate_context(snapshot, [replace(bar, close='101'), news])
        for field in ('available_at', 'observed_at'):
            altered = replace(news, **{field: END, 'ingested_at': END, 'first_seen_at': END if field == 'observed_at' else T})
            selected = replace(snapshot, selected_record_ids=(altered.record_id,), selected_record_hashes=(altered.hash,))
            with self.assertRaises(ValueError):
                validate_context(selected, [altered])
        unknown = replace(news, available_at=None, availability_basis='unknown')
        selected = replace(snapshot, selected_record_ids=(unknown.record_id,), selected_record_hashes=(unknown.hash,))
        with self.assertRaises(ValueError):
            validate_context(selected, [unknown])

    def test_historical_reconstruction_requires_evidence_and_caveats(self):
        snapshot = self.e['ContextSnapshot']
        with self.assertRaises(ValueError):
            replace(snapshot, as_of_mode='historical_source_as_of')
        news = replace(self.e['NewsEvent'], observed_at=END, ingested_at=END, first_seen_at=END,
                       availability_basis='evidenced_historical')
        snapshot = replace(snapshot, as_of_mode='historical_source_as_of', caveats=('Reconstructed later; revisions may be missing.',),
                           selected_record_ids=(news.record_id,), selected_record_hashes=(news.hash,))
        validate_context(snapshot, [news])
        news = replace(news, availability_basis='collector_observation_proxy', available_at=END)
        with self.assertRaises(ValueError):
            validate_context(replace(snapshot, selected_record_hashes=(news.hash,)), [news])

    def test_completed_bars_and_quote_quality(self):
        bar = self.e['PriceBar']
        with self.assertRaises(ValueError):
            replace(bar, available_at='2025-01-06T14:29:00Z')
        with self.assertRaises(ValueError):
            replace(bar, filled=True)
        with self.assertRaises(ValueError):
            replace(self.e['Quote'], bid='102')
        crossed = replace(self.e['Quote'], bid='102', crossed=True)
        snapshot = replace(self.e['ContextSnapshot'], selected_record_ids=(crossed.record_id,), selected_record_hashes=(crossed.hash,))
        with self.assertRaises(ValueError):
            validate_context(snapshot, [crossed])
        stale = replace(self.e['Quote'], stale=True)
        snapshot = replace(snapshot, selected_record_hashes=(stale.hash,))
        with self.assertRaises(ValueError):
            validate_context(snapshot, [stale])
        validate_context(replace(snapshot, stale_record_ids=(stale.record_id,)), [stale])

    def test_future_targets_are_train_only(self):
        target = self.e['FutureTarget']
        for change in ({'split': 'test'}, {'split': 'validation'}, {'stop_gradient': False}):
            with self.assertRaises(ValueError):
                replace(target, **change)

    def test_replay_hash_is_stable_across_process_and_immutable(self):
        record = self.e['NewsEvent']
        raw = record.to_dict()
        constructed = NewsEvent(**raw)
        raw['entities'][0]['confidence'] = '0'
        self.assertEqual(constructed.hash, record.hash)
        with self.assertRaises(FrozenInstanceError):
            constructed.title = 'changed'
        code = 'from cognitive_runtime.adapters.finance.testing.examples import examples; print(examples()["NewsEvent"].hash)'
        result = subprocess.check_output([sys.executable, '-c', code], cwd=ROOT, text=True)
        self.assertEqual(result.strip(), record.hash)
        with self.assertRaises(ValueError):
            NewsEvent.from_json('{"record_id":"a","record_id":"b"}')
        self.assertNotEqual(replace(record, observed_at=END, ingested_at=END, first_seen_at=END).hash, record.hash)

    def test_migrations_are_explicit_copy_on_write_and_breaking_fail_closed(self):
        from cognitive_runtime.adapters.finance.schemas.migrations import migrate_legacy_0_9
        raw = json.loads((FIXTURES / 'prior-0.9.json').read_text())
        before = json.dumps(raw, sort_keys=True)
        with self.assertRaises(ValueError):
            parse_record(raw)
        migrated = migrate_legacy_0_9(raw)
        self.assertEqual(migrated.availability_confidence, '0')
        self.assertEqual(json.dumps(raw, sort_keys=True), before)
        self.assertNotEqual(digest(raw), migrated.hash)
        for version in ('0.8.0', '1.1.0', '2.0.0'):
            with self.assertRaises(ValueError):
                migrate_legacy_0_9({**raw, 'schema_version': version})
        with self.assertRaises(TypeError):
            migrate_legacy_0_9({**raw, 'new_semantics': True})

    def test_committed_fixture_manifest_and_regeneration(self):
        manifest = FixtureManifest.from_json((FIXTURES / 'manifest.json').read_text())
        import jsonschema
        jsonschema.validate(manifest.to_dict(), schema_for(FixtureManifest))
        for entry in manifest.files:
            self.assertEqual(hashlib.sha256((FIXTURES / entry.path).read_bytes()).hexdigest(), entry.sha256)
        with tempfile.TemporaryDirectory() as temp:
            rebuilt = write_fixtures(Path(temp))
            self.assertEqual(rebuilt, manifest)
            for file in Path(temp).iterdir():
                self.assertEqual(file.read_bytes(), (FIXTURES / file.name).read_bytes())
        self.assertLess(sum(f.stat().st_size for f in FIXTURES.iterdir()), 150000)

    def test_synthetic_signal_and_null_records_are_real_contracts(self):
        for null in (False, True):
            payload = generate(zero_signal=null)
            self.assertEqual(payload, generate(zero_signal=null))
            self.assertNotEqual(payload, generate(seed=287, zero_signal=null))
            for record in payload['records']:
                parse_record(record)
            self.assertEqual(len(payload['records']), 24)
            self.assertTrue(payload['synthetic'])
            self.assertIn('training_only_oracle', payload)
        slow = generate(lag=3)
        news = [r for r in slow['records'] if r['schema_name'] == 'NewsEvent'][0]
        self.assertEqual((timestamp(news['available_at']) - timestamp(news['event_at'])).total_seconds(), 180)

    def test_handcrafted_edge_cases_contain_concrete_evidence(self):
        cases = json.loads((FIXTURES / 'edge-cases.json').read_text())
        for case in cases.values():
            for raw in case['records']:
                (ContractTerminalMetadata(**raw) if raw['schema_name'] == 'ContractTerminalMetadata' else parse_record(raw))
        news, delayed, revision = cases['delayed_revised_news']['records']
        self.assertGreater(timestamp(delayed['available_at']), timestamp(T))
        self.assertEqual(revision['supersedes_record_id'], news['record_id'])
        self.assertEqual(cases['shuffled_arrival']['records'], [revision, delayed, news])
        original, repost = cases['syndication']['records']
        self.assertEqual(original['dedup_cluster_id'], repost['dedup_cluster_id'])
        self.assertGreater(timestamp(repost['available_at']), timestamp(original['available_at']))
        skew = cases['clock_skew']['records'][0]
        self.assertGreater(timestamp(skew['event_at']), timestamp(skew['observed_at']))
        old, new = cases['alias_reuse']['records']
        self.assertEqual(old['alias'], new['alias'])
        self.assertNotEqual(old['security_id'], new['security_id'])
        self.assertEqual(old['effective_to'], new['effective_from'])
        split, dividend = cases['split_dividend']['records']
        self.assertEqual(split['split_ratio'], '2')
        self.assertEqual(dividend['cash'], '1')
        self.assertEqual([r['kind'] for r in cases['delisting_halt']['records']], ['delisting', 'halt'])
        stale, crossed, outage = cases['stale_crossed_outage']['records']
        self.assertTrue(stale['stale'] and crossed['crossed'] and outage['missing'])
        closed, terminal = cases['early_closure_terminal_trap']['records']
        self.assertEqual(closed['lifecycle'], 'closed')
        self.assertLess(timestamp(terminal['actual_closed_at']), timestamp(closed['scheduled_expiry_at']))
        calendar = cases['calendar_boundaries']['sessions']
        self.assertIsNone(calendar[0]['open_at'])
        self.assertEqual((timestamp(calendar[1]['close_at']) - timestamp(calendar[1]['open_at'])).total_seconds(), 12600)
        self.assertNotEqual(timestamp(calendar[2]['open_at']).hour, timestamp(calendar[3]['open_at']).hour)

    def test_inference_import_graph_has_no_labels_training_or_optional_frameworks(self):
        code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','transformers','sentence_transformers','alpaca','openai'} or fullname == 'cognitive_runtime.adapters.finance.labels' or fullname.startswith('cognitive_runtime.training'):
   raise AssertionError('forbidden inference import: ' + fullname)
sys.meta_path.insert(0, Block())
import cognitive_runtime.adapters.finance.schemas
import cognitive_runtime.adapters.finance.inference
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
