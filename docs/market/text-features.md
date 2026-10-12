# Snapshot-bound text features (#292)

**Current readiness: synthetic_only baseline; optional model pins verified remotely.**
The checked-in `config/market/models.lock.json` records immutable revisions,
artifact/tokenizer SHA256 and sizes, and selected-revision license evidence.
Normal official metadata access succeeded on 2026-10-12 after the environment
network update. No weights were downloaded or staged: optional capabilities now
fail as `missing_local_artifacts` until the exact approved inventory is present.
Real-model execution, quality and resource fit remain unverified.

## Baseline operation

Use a #291 `Corpus` and validated `SplitPlan`:

```python
from cognitive_runtime.adapters.finance.text.features import TextFeatures

features = TextFeatures('tfidf', dimensions=256).fit(corpus, plan)
train = features.transform(corpus.feature(plan.train[0]))
test = features.transform(corpus.feature(plan.test[0]))
manifest = test.to_dict()
```

`hashing` is the default mode: stable unsigned SHA256 bins of ASCII word counts.
`tfidf` learns document frequency and vocabulary from **train text only** (unique
record hashes, frequency-descending then lexical tie order). IDF is
`log((1+n_documents)/(1+document_frequency))+1`; raw counts times IDF are L2
normalized. Hashing has no learned vocabulary but still requires the validated
train partition to bind fit provenance. No test/validation text, outcome labels or
future-target store is read by fit. Create a new transformer to refit; `state`
returns a copy for provenance, not a mutable live estimator. Mode/dimensions are
read-only; local model identity fields are read-only and pin/provenance accessors
return defensive copies.

Only news IDs and content matching the context snapshot are consumed. Knowledge
clocks are rechecked, deleted news is omitted, and missing text is explicit as
zero rows. Records are sorted by ID. Numeric observations remain structured in
the corpus and are not converted into prose. Numeric/supervised predictors are
#293 scope. These features are not a trained forecasting model.

Preprocessing uses at most 4,096 Unicode code points of `title + LF + body` per
record and 128 text records per snapshot. Oversized snapshots fail rather than
silently select a biased subset. Fit permits 4,096 train samples, 4,096 unique
train text records and a 65,536-term intermediate vocabulary; dimensions are
1–4,096. Every output declares selected IDs/content hashes, snapshot hash,
truncation flags, float32 values, dimensions, feature signature and fit provenance.
Empty/unknown-word vectors remain zero. Nonfinite or wrong-shaped encodings fail.

`FrozenTextFeatures(encoder)` shares frozen features with supervised and future
JEPA consumers. Trainable context/target projections, loss and EMA are deliberately
not part of this backbone artifact. Future targets must still enter through the
separate #291 training capability; this API never reads them.

## Cache and retention

`FeatureCache` is process-local LRU: default 256 entries, hard ceiling 1,024,
maximum 128 KiB canonical serialized value per entry. Content is serialized on
insert to avoid mutable aliases. Keys bind record ID, revision and full record
hash plus model, tokenizer and preprocessing identity. Extractor preprocessing
also binds prompt and schema hashes. Cache corruption is not a trusted bypass:
returned vectors and extracted outputs are validated again. `clear()` drops local
references; process exit releases them. No disk cache or retention policy is
silently introduced. Exported artifacts, operator copies and source revocation
remain the operator's rights/retention responsibility.

## Optional local models: verified pins, explicit staging

| Model | Immutable model/tokenizer revision | Selected-revision license evidence |
| --- | --- | --- |
| MiniLM | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | [README](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md) declares Apache-2.0; upstream has no separate LICENSE |
| Qwen | `7ae557604adf67be50417f59c2c2f167def9a775` | [README](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/README.md) and [LICENSE](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE) agree on Apache-2.0 |

MiniLM uses 384 dimensions, 256-token truncation, attention-mask mean pooling,
L2 normalization, CPU float32 and frozen parameters, matching its card's
Transformers approach. Qwen uses CPU float32 rather than the upstream bfloat16
configuration; it remains optional and may not fit our hard resource limits.
Neither selected card substantiates a complete training cutoff.

Each lock entry provides the official revision metadata URL and immutable file
URLs. Weight SHA256/size come from that revision's official LFS metadata:
**weight bytes have not been downloaded or locally hashed**. Every other listed
file was downloaded, matched to metadata by byte size and Git blob SHA1, then
SHA256-hashed locally. `status: verified` means reviewed pin evidence, not staged
weights or successful inference. Runtime rehashes every staged file before use.

Stage the exact `local.REQUIRED` inventory plus README for both models and
LICENSE for Qwen only. The MiniLM README is its pinned license evidence; no
fabricated LICENSE file is required. Alternate pickle weights, custom Python,
extra configs and symlinks are rejected. All file evidence URLs must name the
same immutable model/tokenizer revision. Changes require review; setting
`status` alone cannot pass the validator. Do not bypass denied access or download
weights automatically. The repository contains metadata hashes, not model files.

To opt in, explicitly install `.[market-text]` and stage exactly the
approved files in a private directory. The library has **no downloader**:

```python
from cognitive_runtime.adapters.finance.text.local import LocalModel
from cognitive_runtime.adapters.finance.text.features import FrozenTextFeatures
from cognitive_runtime.adapters.finance.text.extractor import StructuredExtractor

encoder = LocalModel('minilm', lock_path='config/market/models.lock.json', staged=local_minilm_dir)
encoded = FrozenTextFeatures(encoder).transform(feature)
extractor = LocalModel('qwen', lock_path='config/market/models.lock.json', staged=local_qwen_dir)
claims = StructuredExtractor(extractor).transform(feature)
```

Missing/unapproved pins, missing/mismatched files or missing dependencies fail
clearly, before optional imports. Artifact and request/pin validation is repeated inside the child. The shared
parent/child request ceiling is 256 KiB of UTF-8 serialized JSON, including
escape expansion; oversized payloads fail before model imports. Only `local_files_only=True`, `trust_remote_code=False` and safetensors
loading are allowed. A Linux child enforces 4 GiB virtual address space, 50 CPU
seconds and two threads, with a parent 60-second wall timeout per invocation.
MiniLM batches contain at most eight 256-token sequences. Qwen uses one input,
a 1,536-token prompt ceiling and at most 384 generated tokens, deterministic greedy
decoding and no tools. Each child loads then exits; reload overhead is intentional.
Unsupported enforcement fails closed. This may make a particular real model
unavailable on a host; do not silently relax limits to obtain a pass.

Offline environment flags and an audit hook deny/record socket operations; even a
caught network exception invalidates child success. This is defense in depth for
trusted local libraries, **not an adversarial native-code sandbox**. Model files
must remain operator-controlled and unchanged during use. Explicit smoke should
also run in an OS-level no-network environment and record runtime/RSS; mocks are
not evidence of real-model performance or offline compliance.

## Extraction semantics

Qwen receives untrusted source data and a versioned prompt/schema. Generation is
not grammar constrained; **only strict postvalidated JSON is returned**, and
invalid generation fails without repair, retry, tools or remote fallback. Output
is at most eight classifications with enum kind/direction/uncertainty and exact
Unicode evidence offsets/text. Extra fields, instructions as classifications,
duplicate keys, wrong types, wrong spans and unsupported uncertainty fail.
Evidence strings remain untrusted data and must never be interpreted as actions.
A valid span proves substring identity, not semantic entailment. The result is
explicitly model-derived, not an authoritative fact or calibrated confidence.
An injected source can affect model opinion within the schema; it cannot add a
tool channel or unvalidated fields.

Both candidate models have **unknown substantiated training cutoff** in this
implementation. Historical contamination/memorization remains possible. Model
hashes establish identity only; compare the non-pretrained baseline and
prospective/post-release holdouts. Name masking is not proof of leakage freedom.
No live financial validation, forecast gain or tradable-edge claim is made.

## Validation

```sh
python -m pytest tests/market/test_text_features.py -q
# Explicit opt-in, pre-staged weights only; fail rather than skip missing inputs:
python -m pytest tests/market/text/test_local_model_smoke.py --run-market=text -k minilm -s
python -m pytest tests/market/text/test_local_model_smoke.py --run-market=text -k qwen -s
```

Default subprocess tests exercise the real worker shell with only inference
replaced by a synthetic function: byte framing, hard resource limits, artifact
rechecks and caught synthetic audit-denial accounting. No socket is opened by
the synthetic audit probe. This does not validate actual ML execution.

Default collection does not import the optional tier. **Neither real-model smoke
was run**: staged weights are unavailable. Synthetic fake tests
cover schema/cache/import boundaries; they do not validate real tokenization,
loading, extraction usefulness or that the real model fits resource limits.
