# Phase 0 — Identity & Scaffolding

> Master plan: [Phase 0](../02-implementation-plan.md#phase-0--identity--scaffolding-small-unblocks-everything).
> **Goal:** the organism has a configurable `name`, and that name threads through
> every generated artifact (checkpoints, session ids, exports, dashboard grouping).
> This is deliberately small — cheap provenance that is easiest to add *before*
> there are files to rename.

## Why first

Provenance is trivial to add now and painful to retrofit once files carry
organism-specific names. Everything the later phases generate (dream exports,
consolidation checkpoints, clinic session groups) keys off the organism name, so
it has to exist before those artifacts do.

**Explicitly deferred to after Milestone 2:** the package-tree rename
(`organism/`, `world/`, `brain/`, `sleep/`, …) and `ARCHITECTURE_MAP.md`. Do not
create empty re-export namespaces now — that is renaming ~100 files before the
three missing organs exist. Phase 0 is *only* the `name`.

## Dependencies

None. This phase unblocks everything else.

## Builds on (existing code)

- `cognitive_runtime/runtime/config.py` — `RuntimeConfig` dataclass (`session_id`,
  `record_dir`, `curriculum` already live here).
- `cognitive_runtime/runtime/recorder.py` — writes session dirs and metadata.
- `cognitive_runtime/neural/checkpoint.py` — `NeuralAgentCheckpoint` bundle +
  `checkpoint_metadata()` on models.
- `cognitive_runtime/cli.py` — `cmd_run`, `cmd_nursery_run`, `cmd_train`,
  `cmd_curriculum_run` and their argparse setup (`build_parser` around line 1669).
- `cognitive_runtime/tools/metrics_dashboard.py` — groups sessions for reporting.

## Tasks

1. **Add the organism name to config.**
   - Add `name: Optional[str] = None` to `RuntimeConfig` (`runtime/config.py`).
   - When `None`, generate a stable default (e.g. a short readable slug like
     `sprout-7f3a`) once at run start and freeze it into the resolved config, so a
     run always records a concrete name, never `None`.
   - *Acceptance:* `RuntimeConfig(name="Pixel")` round-trips; an unset run resolves
     to a generated name recorded in session metadata.

2. **Thread `name` into the session id / record path.**
   - In `recorder.py` (and wherever `session_id` is derived), prefix the session id
     with the organism name: `Pixel-<timestamp-or-uuid>`.
   - Record `name` as a top-level field in session metadata JSON.
   - *Acceptance:* a run produces `sessions/Pixel-<id>/` and its metadata contains
     `"name": "Pixel"`.

3. **Stamp `name` into checkpoint metadata.**
   - Add `name` to the checkpoint bundle metadata in `neural/checkpoint.py` so a
     loaded checkpoint knows which organism it belongs to. Keep it optional on load
     for backward compatibility (old checkpoints have no name → treat as legacy).
   - *Acceptance:* saving then loading a checkpoint preserves `name`; a nameless
     legacy checkpoint still loads.

4. **Thread `name` into export filenames.**
   - `viewer/export_predictions.py` / `training/prediction_export.py`: prefix dream/
     prediction export files with the organism name.
   - *Acceptance:* an export lands as `Pixel-<...>.json` (or the existing format,
     name-prefixed).

5. **Wire `--name` through the CLI.**
   - Add `--name` to the shared run arguments (`run`, `nursery run`, `train`,
     `curriculum run`) in `cli.py`. Default: unset → generated.
   - *Acceptance:* `... run --name Pixel` records under `Pixel-<id>`; omitting it
     records under the generated name.

6. **Group by name in the dashboard.**
   - `tools/metrics_dashboard.py`: allow grouping/filtering recorded sessions by
     organism `name` (falling back to "legacy" for nameless sessions).
   - *Acceptance:* the dashboard can list all sessions for a given organism name.

## Deliverables

- `RuntimeConfig.name`, resolved-never-`None`.
- Name-prefixed session ids, name in session + checkpoint metadata, name-prefixed
  exports, `--name` CLI flag, name-grouped dashboard.

## Tests

- `tests/test_program.py` / a new `tests/test_organism_name.py`:
  - config round-trip with and without an explicit name;
  - a recorded run's session dir + metadata carry the name;
  - checkpoint save/load preserves the name and still loads a legacy (nameless)
    checkpoint.

## Milestone 0 (exit gate)

A run recorded as `Pixel-<session>`, its checkpoint carrying `name: Pixel`, and
**every generated file discoverable by organism name**. Verified by the tests
above plus a manual `run --name Pixel` producing a fully name-tagged session dir,
checkpoint, and export.

## Risks / notes

- Keep the name **cosmetic to behaviour**: it must not change any learning or
  recording semantics, only identifiers. A run with `--name X` and one with
  `--name Y` on the same seed should differ *only* in file names/ids.
- Backward compatibility: every place that reads a name must tolerate its absence
  (legacy sessions/checkpoints predate this field).
