"""Deterministic, cosmetic display names for Model Factory runs.

``run_id`` and the factory contracts are the identities of a run.  This
module deliberately only derives a friendly label after those identities have
already been allocated; no value produced here is suitable for seeding
training, locating artifacts, or reconstructing lineage.
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import namesgenerator


DISPLAY_NAME_FORMAT = "model-factory-display-name-v1"
SHORT_RUN_ID_LENGTH = 8


@dataclass(frozen=True)
class DisplayNameParent:
    """The already-assigned cosmetic name of a known lineage parent."""

    run_id: str
    display_name: str


@dataclass(frozen=True)
class DisplayNameAssignment:
    """The immutable display-name record written beside a run's artifacts."""

    display_name: str
    naming_seed: str
    surname_source_run_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "format": DISPLAY_NAME_FORMAT,
            "display_name": self.display_name,
            "naming_seed": self.naming_seed,
            "surname_source_run_ids": list(self.surname_source_run_ids),
        }


def new_naming_seed() -> str:
    """Return a persisted seed for a cosmetic name, independent of training."""

    return secrets.token_hex(16)


def surname_sequence(display_name: str) -> tuple[str, ...]:
    """Return a display name's surname parts, excluding its generated first part."""

    parts = tuple(part for part in display_name.split("-") if part)
    if len(parts) < 2:
        raise ValueError(f"display name needs a generated first part and surname: {display_name!r}")
    return parts[1:]


def _parent_surnames(parent: DisplayNameParent) -> tuple[str, ...]:
    """Read cosmetic surname parts without inheriting a collision suffix."""

    collision_suffix = parent.run_id[-SHORT_RUN_ID_LENGTH:]
    # A run ID's trailing eight characters can themselves contain a hyphen
    # (for example ``1-8fc9b9``).  Remove the serialized suffix before
    # splitting the cosmetic name, rather than trying to match one token.
    display_name = parent.display_name
    serialized_suffix = f"-{collision_suffix}"
    if display_name.endswith(serialized_suffix):
        display_name = display_name[:-len(serialized_suffix)]
    return surname_sequence(display_name)


def _random_for_seed(naming_seed: Union[str, int]) -> random.Random:
    # Python's hash randomisation would make ``hash(seed)`` process-dependent.
    # A SHA-256-derived integer is stable for both numeric and textual seeds.
    encoded = str(naming_seed).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(encoded).digest(), "big"))


def _generated_parts(rng: random.Random) -> tuple[str, str]:
    """Choose from namesgenerator's own adjective and surname vocabularies."""

    while True:
        first = rng.choice(namesgenerator.left)
        surname = rng.choice(namesgenerator.right)
        # Preserve namesgenerator's one intentional exclusion.
        if (first, surname) != ("boring", "wozniak"):
            return first, surname


def _normalise_existing(existing_names: Union[Iterable[str], Mapping[str, object]]) -> set[str]:
    return set(existing_names)


def assign_display_name(
    *,
    run_id: str,
    naming_seed: Union[str, int],
    parents: Sequence[DisplayNameParent] = (),
    existing_names: Union[Iterable[str], Mapping[str, object]] = (),
) -> DisplayNameAssignment:
    """Assign a deterministic display name without changing any real identity.

    ``parents`` is empty for a fresh trial, contains one parent for a clone or
    fine-tune, and contains the two configuration parents for a genetic child.
    For two parents the seed also fixes their cosmetic ordering.  A collision
    only changes this newly-assigned label by adding a short run-ID suffix;
    it never alters an existing record.
    """

    if len(parents) > 2:
        raise ValueError("a display name accepts at most two configuration parents")
    if not run_id:
        raise ValueError("run_id must not be empty")

    seed = str(naming_seed)
    rng = _random_for_seed(seed)
    first, fresh_surname = _generated_parts(rng)
    sources: tuple[str, ...]
    if not parents:
        surnames = (fresh_surname,)
        sources = ()
    elif len(parents) == 1:
        surnames = _parent_surnames(parents[0])
        sources = (parents[0].run_id,)
    else:
        ordered = list(parents)
        rng.shuffle(ordered)
        surnames = (
            _parent_surnames(ordered[0])[0],
            _parent_surnames(ordered[1])[-1],
        )
        sources = (ordered[0].run_id, ordered[1].run_id)

    name = "-".join((first, *surnames))
    if name in _normalise_existing(existing_names):
        suffix = run_id[-SHORT_RUN_ID_LENGTH:]
        name = f"{name}-{suffix}"
    return DisplayNameAssignment(name, seed, sources)


def load_display_name(path: Union[str, Path]) -> DisplayNameAssignment:
    """Read and validate a persisted display-name record."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != DISPLAY_NAME_FORMAT:
        raise ValueError(f"not a display-name manifest: {path}")
    try:
        sources = payload["surname_source_run_ids"]
        if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
            raise ValueError("surname_source_run_ids must be a list of strings")
        return DisplayNameAssignment(
            display_name=str(payload["display_name"]),
            naming_seed=str(payload["naming_seed"]),
            surname_source_run_ids=tuple(sources),
        )
    except KeyError as exc:
        raise ValueError(f"invalid display-name manifest: {path}") from exc


__all__ = [
    "DISPLAY_NAME_FORMAT",
    "SHORT_RUN_ID_LENGTH",
    "DisplayNameAssignment",
    "DisplayNameParent",
    "assign_display_name",
    "load_display_name",
    "new_naming_seed",
    "surname_sequence",
]
