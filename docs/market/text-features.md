# Snapshot-bound text features (#292)

**Current readiness: synthetic_only baseline; optional model pin resolution blocked.**
The checked-in `config/market/models.lock.json` deliberately contains no invented
commits or SHA256 values. Both optional capabilities fail as `unverified_model_pin`.
The execution environment denied the official model metadata request with proxy
HTTP 403 on 2026-10-12. Do not mark #292 complete until reviewed immutable pins
replace this disabled inventory and the acceptance matrix is reconciled.

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
returns a copy for provenance, not a mutable live estimator.

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

## Optional local models: blocked pending verified pins

Candidate official cards:

- [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) describes
  Apache-2.0, 384-dimensional sentence embeddings and 256-wordpiece truncation.
  Selected runtime settings are attention-mask mean pooling, L2 normalization,
  CPU float32 and frozen parameters. We use the card's Transformers approach,
  not a mandatory sentence-transformers import.
- [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) is the optional local
  extraction candidate. License **at the selected immutable revision is still
  unverified**; the runtime accepts only reviewed Apache-2.0 pins for this candidate.

These current card reads do not establish immutable artifact identity. A completed
lock needs a 40-hex model commit and matching tokenizer revision, exact SHA256 and
byte size of **every loaded file**, immutable official evidence URLs, and reviewed
README/license hashes. Use the exact file inventory in `local.REQUIRED` plus
`README.md` and `LICENSE`; no alternate pickle weights, custom Python code, extra
configs or symlinks are allowed. Resolve metadata through permitted official
sources; never fabricate hashes, bypass denied access, or silently download.
The repository currently ships only the disabled lock. A future verified lock
requires code review; setting `status` alone cannot pass the validator.

Once pins are reviewed, explicitly install `.[market-text]` and stage exactly the
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
clearly, before optional imports. Artifact verification is repeated inside the
child. Only `local_files_only=True`, `trust_remote_code=False` and safetensors
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

Default collection does not import the optional tier. **Neither real-model smoke
was run**: pins/access and staged weights are unavailable. Synthetic fake tests
cover schema/cache/import boundaries; they do not validate real tokenization,
loading, extraction usefulness or that the real model fits resource limits.
