"""Explicit bounded two-seed synthetic evaluation; MIT project-authored seed 294."""
import pytest
from tests.market.test_distributional_baselines import corpus
from cognitive_runtime.adapters.finance.evaluation.ablations import run_suite, Budget, ARMS

pytestmark = pytest.mark.market_extended


def test_two_seed_equal_budget(corpus):
    c, plan = corpus
    reports = run_suite(c, plan, listing_id='contract:demo', population='synthetic',
                        budget=Budget(seeds=(294, 295), text_dimensions=4), bootstrap={'repetitions': 40})
    assert len(reports) == 2 * len(ARMS)
    assert {r['ablation']['seed'] for r in reports} == {294, 295}
    assert all(r['summary']['counts']['samples'] == len(plan.validation) for r in reports)
    assert len({r['identities']['split_sha256'] for r in reports}) == 1
    assert all(not r['edge_promotable'] for r in reports)
