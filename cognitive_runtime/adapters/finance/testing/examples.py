"""Project-authored synthetic examples, shared by docs, schemas and fixture tests."""
from dataclasses import replace
from cognitive_runtime.adapters.finance.schemas import *
from cognitive_runtime.adapters.finance.schemas.records import ContextWindow, Quantile, EntityEvidence
from cognitive_runtime.adapters.finance.labels import LabelSpec, RealizedLabel, ContractTerminalMetadata, FutureTarget

T = '2025-01-06T14:30:00Z'
END = '2025-01-06T15:30:00Z'
H = '0' * 64


def envelope(kind, identity=None, at=T):
    return dict(schema_name=kind.__name__, schema_version=VERSION, record_id=identity or kind.__name__ + ':demo:r1',
                source_id='synthetic', source_record_id=identity or kind.__name__ + ':demo', revision=1,
                supersedes_record_id=None, event_at=at, available_at=at, observed_at=at, ingested_at=at,
                availability_basis='synthetic_ground_truth', availability_confidence='1',
                original_timezone='UTC', timestamp_precision='second',
                provenance=Provenance(raw_sha256=H, adapter_version='synthetic-v1', rights='project-authored-fixture',
                                      license='MIT', synthetic=True, untrusted_source_text=True))


def examples():
    instrument = Instrument(**envelope(Instrument), issuer_id='issuer:demo', security_id='security:demo',
                            listing_id='listing:demo', share_class='common', venue='SYNTH', currency='USD', kind='equity',
                            successor_security_id=None, external_ids=('synthetic:demo',), effective_from=T,
                            effective_to=None, universe_id='synthetic-universe-v1')
    alias = InstrumentAlias(**envelope(InstrumentAlias), alias='DEMO', namespace='ticker:SYNTH', security_id='security:demo',
                            listing_id='listing:demo', effective_from=T, effective_to=None)
    bar = PriceBar(**envelope(PriceBar), listing_id='listing:demo', venue='SYNTH', feed='synthetic', currency='USD',
                   mark=Mark.TRADE, start_at='2025-01-06T14:29:00Z', end_at=T, session_id='synthetic:2025-01-06',
                   calendar_version='synthetic-v1', adjustment='raw', open='100', high='101', low='99', close='100',
                   volume='10', volume_unit='shares', complete=True, stale=False, missing=False, filled=False)
    quote = Quote(**envelope(Quote), listing_id='listing:demo', venue='SYNTH', feed='synthetic', currency='USD',
                  bid='99', ask='101', stale=False, missing=False, crossed=False)
    action = CorporateAction(**envelope(CorporateAction), security_id='security:demo', kind='split', announced_at=T,
                             effective_at=END, ex_at=END, record_at=None, payment_at=None, split_ratio='2', cash=None,
                             currency=None, successor_security_id=None, evidence_record_ids=('announcement:demo',))
    news = NewsEvent(**envelope(NewsEvent), original_published_at=T, first_seen_at=T,
                     title='Fictional production report', text='Fictional Alpha reports stable production.', links=(),
                     entities=(EntityEvidence(issuer_id='issuer:demo', instrument_ids=('security:demo',), confidence='1',
                                              ambiguous=False, supporting_span='Fictional Alpha'),),
                     dedup_cluster_id='story:demo', deleted=False, extraction_model_id=None)
    contract = PredictionContractState(**envelope(PredictionContractState), market_id='market:demo', token_id='token:yes',
                                       listing_id='contract:demo', side='YES', question='Synthetic signal positive?',
                                       question_version='1', payout_scale='1', currency='USD', quote_basis=Mark.MID,
                                       lifecycle='open', scheduled_expiry_at='2025-01-07T14:30:00Z', probability_price='0.42')
    window = ContextWindow(clock='exchange_sessions', count=20, calendar_id='synthetic', calendar_version='1')
    snapshot = ContextSnapshot(**envelope(ContextSnapshot), decision_at=T, as_of_mode='strict_replay',
                               selected_record_ids=(bar.record_id, news.record_id), selected_record_hashes=(bar.hash, news.hash),
                               price_context=window, text_context=ContextWindow(clock='utc_elapsed_seconds', count=259200,
                               calendar_id=None, calendar_version=None), missing_record_ids=(), stale_record_ids=(),
                               feature_schema='market-numeric-v1', preprocessing_sha256=H, manifest_sha256=H, caveats=())
    horizon = Horizon(clock=Clock.ELAPSED, count=3600, calendar_id=None, calendar_version=None)
    spec = LabelSpec(**envelope(LabelSpec), target=Target.CONTRACT_CHANGE, unit=Unit.PRICE_CHANGE, horizon=horizon,
                     currency='USD', feed='synthetic', mark=Mark.MID, anchor_policy='last_eligible_at_or_before_decision',
                     anchor_max_age_seconds=60, endpoint_policy='first_at_or_after', endpoint_tolerance_seconds=60,
                     action_policy='none', terminal_policy='censor_without_preterminal_market_price')
    label = RealizedLabel(**envelope(RealizedLabel, at=END), spec=spec, listing_id='contract:demo', decision_at=T,
                          anchor_record_id=contract.record_id, endpoint_record_id='contract:demo:end', action_record_ids=(),
                          anchor_price='0.42', endpoint_price='0.47', anchor_at=T, nominal_endpoint_at=END, actual_endpoint_at=END,
                          anchor_currency='USD', endpoint_currency='USD', anchor_feed='synthetic', endpoint_feed='synthetic',
                          anchor_mark=Mark.MID, endpoint_mark=Mark.MID, split_multiplier='1', value='0.05',
                          target_available_at=END, censor_reason=None)
    forecast = ForecastDistribution(**envelope(ForecastDistribution), snapshot_id=snapshot.record_id, listing_id='contract:demo',
                                    decision_at=T, target=Target.CONTRACT_CHANGE, unit=Unit.PRICE_CHANGE, horizon=horizon,
                                    anchor_price='0.42', anchor_at=T, anchor_record_id=contract.record_id, currency='USD',
                                    feed='synthetic', mark=Mark.MID, support_lower='-0.42', support_upper='0.58',
                                    quantiles=(Quantile(level='0.1', value='-0.1'), Quantile(level='0.5', value='0'),
                                               Quantile(level='0.9', value='0.2')), direction=None, model_id='synthetic-example',
                                    calibrator_id=None, abstention_reason=None, quality=('synthetic_only',))
    provider = ProviderCapabilityManifest(**envelope(ProviderCapabilityManifest), provider='synthetic', status='synthetic_only',
                                          authentication='none', free_entitlement='project-authored', usage_rights='MIT',
                                          access='offline', endpoint_allowlist=(), requests_per_minute=None, latency_seconds=0,
                                          timestamp_quality='ground_truth', revision_quality='ground_truth', history_quality='tiny',
                                          identity_quality='synthetic', evidence_at=T, evidence_urls=(), provider_budget_usd='0')
    run = RunManifest(**envelope(RunManifest), **{name + '_sha256': H for name in
                      ('code', 'config', 'environment', 'data', 'model', 'calendar', 'split', 'rights')},
                      status='completed', failure=None, provider_budget_usd='0', runtime_seconds='0')
    report = EvaluationReport(**envelope(EvaluationReport), run_id=run.record_id, run_sha256=run.hash, dataset_kind='synthetic',
                              forecast_ids=(), metrics=(), total_count=0, evaluated_count=0, censored_count=0,
                              abstained_count=0, stale_count=0, caveats=('Schema example; no forecasting evaluation performed.',))
    terminal = ContractTerminalMetadata(**envelope(ContractTerminalMetadata, at=END), market_id='market:demo', token_id='token:yes',
                                        resolved_at=END, actual_closed_at=END, payout='1', currency='USD', outcome='YES')
    future = FutureTarget(**envelope(FutureTarget, at=END), split='train', context_snapshot_id=snapshot.record_id,
                          window_start_at=T, window_end_at=END, future_record_ids=('future:synthetic',), stop_gradient=True)
    return {type(x).__name__: x for x in (instrument, alias, bar, quote, action, news, contract, snapshot, spec, label,
                                         forecast, provider, run, report, terminal, future)}
