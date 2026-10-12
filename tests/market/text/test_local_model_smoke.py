"""Explicit pre-staged CPU smoke; no downloads and no implicit skips on selection."""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.market_text


def test_local_minilm_smoke():
    from cognitive_runtime.adapters.finance.text.local import LocalModel
    root = os.environ.get('CCR_MINILM_STAGED')
    if not root:
        pytest.fail('CCR_MINILM_STAGED must point to the exact reviewed file inventory')
    model = LocalModel('minilm', lock_path=Path('config/market/models.lock.json'), staged=root)
    rows = model.encode(['A fictional issuer reports stable production.', 'Synthetic zero-signal control.'])
    assert len(rows) == 2 and all(len(row) == 384 for row in rows)
    assert all(abs(sum(v*v for v in row)-1) < 1e-4 for row in rows)
    print(json.dumps({'model':'minilm','metrics':model.last_metrics,'signature':model.signature}))


def test_local_qwen_smoke():
    from cognitive_runtime.adapters.finance.text.local import LocalModel
    from cognitive_runtime.adapters.finance.text.extractor import validate_output
    root = os.environ.get('CCR_QWEN_STAGED')
    if not root:
        pytest.fail('CCR_QWEN_STAGED must point to the exact reviewed file inventory')
    model = LocalModel('qwen', lock_path=Path('config/market/models.lock.json'), staged=root)
    text = 'Synthetic issuer reports a production increase; the report is unverified.'
    validate_output(model.generate(text), text)
    print(json.dumps({'model':'qwen','metrics':model.last_metrics,'signature':model.signature}))
