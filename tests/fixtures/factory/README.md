# Synthetic legacy Model Factory goldens

These small artifacts restore the checked-in compatibility gate for
[Factory #212](https://github.com/Brain-Crumbs/CCR/issues/212) and
[lightweight CI #297](https://github.com/Brain-Crumbs/CCR/issues/297).

**Status: `synthetic_only`.** All inputs are original project-authored synthetic
content, contributed under the repository's MIT license. No external dataset,
market/news payload, credential, trained model or checkpoint is included.
Checkpoint/session hashes in the synthetic configuration are SHA-256 identifiers
of project-authored seed strings, not claims that real source files were read.

- `contracts.json`: small architecture, nonempty disjoint data split identities,
  resolved optimizer configuration, and their real canonical contract hashes
- `lineage.fresh.json` / `lineage.clone.json`: absent parent versus a checkpoint
  donor and two configuration parents
- `state.queued.json` / `state.completed.json`: nullable initial fields versus
  an append-only four-step lifecycle with deterministic timestamps

The three schemas close the artifact and fixed contract objects, require their
version and fields, and constrain hashes, types, bounds, lifecycle states and
parent presence. Intentionally extensible policy and optimizer settings remain
mappings, consistent with the production contract's historical read support.
They are representative compatibility schemas, not a claim to cover every
Factory report or replace runtime validation.

## Reproduce and verify

With the base + dev environment installed, from the repository root:

```sh
python tests/fixtures/factory/metadata/generate.py
python .github/scripts/validate_artifact_schemas.py
python -m pytest tests/test_factory_artifact_schemas.py -q
```

The generator uses seed `297` and reviewed inputs in `metadata/inputs.json`.
It invokes the actual spec resolver, contract constructors, run allocator and
state transitions. Only state clocks and irrelevant host execution snapshots
are replaced. It never trains, loads a checkpoint or downloads data. Tests also
load committed state files through `load_state`, reconstruct each contract, and
assert canonical hashes and regenerated artifact bytes match the goldens.

`metadata/manifest.json` records source, seed, MIT rights, intended edge cases,
and SHA-256 hashes of every artifact, its schema, the input configuration and
the generator. Metadata stays in its own directory so it cannot accidentally
be treated as an artifact by the validator's top-level JSON glob. Hashes are
review evidence, not signatures; regenerated changes still require review.

When an intentional format change occurs, update the schemas/inputs, regenerate,
and review both byte changes and hash changes. Never replace a schema with an
empty object or skip validation because an input directory is absent. The
standalone validator also checks the provenance inventory and all recorded hashes.
It rejects missing baseline pairs, orphans, malformed JSON/schemas,
duplicate keys, non-finite JSON constants and external schema references.
