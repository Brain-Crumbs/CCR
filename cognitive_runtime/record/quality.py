"""World-agnostic recording-quality gates (issue #90).

Generalizes ``training.nursery``'s Minecraft-only gate (issue #62:
``EpisodeRecordingQuality`` / ``measure_recording_quality`` /
``validate_nursery_recordings``) so it reads any World's stream log the same
way: pixel provenance, motion floor, completed-episode, and a facing-sweep
check that covers both continuous yaw (Minecraft's ``spatial.rotation``) and
discrete facing (Crafter's ``spatial.facing`` -- a ``{x, y}`` grid direction,
flipped on every directional move attempt whether or not it succeeds; see
``programs/crafter/observations.py``). Nothing here depends on which Program
produced the log; it only reads ``*.streams.jsonl`` / ``*.summary.json``.

Adds a green/amber/red verdict per session on top of the boolean pass/fail
gate -- the shape the read-only clinic backend (phase 8) will consume:
red = a hard-fail issue (the recording cannot support the training claim it
exists to make); amber = it clears every floor but only within
``AMBER_MARGIN`` of one, or is missing pixel-provenance metadata (recorded
before that field existed) so provenance can't be confirmed either way;
green = clears every floor with margin to spare.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from cognitive_runtime.runtime.replay import list_episodes

#: How far above a floor (or below a ceiling) counts as "comfortably clear"
#: rather than "amber -- passed, but close to the line".
AMBER_MARGIN = 1.5

_VERDICT_RANK = {"green": 0, "amber": 1, "red": 2}


@dataclass
class EpisodeRecordingQuality:
    """What the gate measures from one recorded episode's stream log."""

    session_dir: str
    episode_id: str
    n_frames: int
    unique_frames: int
    net_displacement: float
    duration_ticks: int
    #: Furthest x/z (or x/y) distance from the episode's starting position --
    #: catches an agent that drifted away and back (net displacement ~0) just
    #: as well as one that walked off.
    max_displacement: float = 0.0
    #: Total |wrapped yaw delta| over the episode, in degrees (continuous
    #: facing -- Minecraft's ``spatial.rotation``).
    yaw_sweep_degrees: float = 0.0
    #: Discrete-facing equivalent (Crafter's ``spatial.facing``): how many
    #: times the facing direction changed, and how many distinct directions
    #: were visited (max 4 on a grid).
    facing_changes: int = 0
    unique_facings: int = 0
    #: ``summary.success`` -- False when the episode terminated early (death);
    #: ``None`` for recordings whose summary predates the field or is absent.
    completed: Optional[bool] = None
    termination_reason: str = ""
    #: Pixel provenance reported by the backend/world (e.g. ``viewer``/
    #: ``grid`` for Minecraft, ``crafter`` for Crafter's native render), empty
    #: for recordings that predate provenance tracking.
    pixel_sources: List[str] = field(default_factory=list)

    @property
    def unique_frame_fraction(self) -> float:
        return self.unique_frames / self.n_frames if self.n_frames else 0.0

    @property
    def blocks_per_tick(self) -> float:
        return self.net_displacement / self.duration_ticks if self.duration_ticks else 0.0

    @property
    def max_blocks_per_tick(self) -> float:
        return self.max_displacement / self.duration_ticks if self.duration_ticks else 0.0


def _wrapped_degrees(delta: float) -> float:
    return abs((delta + 180.0) % 360.0 - 180.0)


def measure_recording_quality(session_dir: str, episode_id: str) -> EpisodeRecordingQuality:
    """Scan one episode's stream log for the gate's signals: unique pixel
    frames (via content-hash ``frame_ref``), x/y displacement (net and max),
    yaw sweep and/or discrete facing changes, episode completion, and pixel
    provenance. World-agnostic: reads whatever subset of
    ``spatial.position``/``spatial.rotation``/``spatial.facing`` the
    recording's world actually publishes."""

    first_pos: Optional[tuple] = None
    last_pos: Optional[tuple] = None
    max_displacement = 0.0
    last_yaw: Optional[float] = None
    yaw_sweep = 0.0
    last_facing: Optional[tuple] = None
    facing_changes = 0
    facings_seen: set = set()
    n_frames = 0
    frame_refs: set = set()
    streams_path = os.path.join(session_dir, f"{episode_id}.streams.jsonl")
    with open(streams_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stream_id = record.get("stream_id")
            if stream_id == "vision.frame.pixels":
                n_frames += 1
                ref = record.get("frame_ref") or record.get("hash")
                if ref:
                    frame_refs.add(ref)
            elif stream_id == "spatial.position":
                payload = record.get("payload") or {}
                # Minecraft's position is 3-D ({x, y, z}, y = height) -- the
                # horizontal plane the gate cares about is x/z. Crafter's is
                # a genuine 2-D grid ({x, y}, no z key), so fall back to y
                # only when z is absent.
                horizontal = payload.get("z", payload.get("y", 0.0))
                pos = (float(payload.get("x", 0.0)), float(horizontal))
                if first_pos is None:
                    first_pos = pos
                else:
                    max_displacement = max(
                        max_displacement,
                        math.hypot(pos[0] - first_pos[0], pos[1] - first_pos[1]),
                    )
                last_pos = pos
            elif stream_id == "spatial.rotation":
                payload = record.get("payload") or {}
                yaw = payload.get("yaw")
                if isinstance(yaw, (int, float)):
                    if last_yaw is not None:
                        yaw_sweep += _wrapped_degrees(float(yaw) - last_yaw)
                    last_yaw = float(yaw)
            elif stream_id == "spatial.facing":
                payload = record.get("payload") or {}
                facing = (payload.get("x"), payload.get("y"))
                facings_seen.add(facing)
                if last_facing is not None and facing != last_facing:
                    facing_changes += 1
                last_facing = facing

    displacement = (
        math.hypot(last_pos[0] - first_pos[0], last_pos[1] - first_pos[1])
        if first_pos is not None and last_pos is not None
        else 0.0
    )
    duration_ticks = 0
    completed: Optional[bool] = None
    termination_reason = ""
    pixel_sources: List[str] = []
    summary_path = os.path.join(session_dir, f"{episode_id}.summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        duration_ticks = int(summary.get("duration_ticks", 0))
        if "success" in summary:
            completed = bool(summary["success"])
        termination_reason = str(summary.get("termination_reason", ""))
        program_stats = summary.get("program_stats") or {}
        sources = program_stats.get("pixel_sources")
        if isinstance(sources, list):
            pixel_sources = [str(s) for s in sources]
    return EpisodeRecordingQuality(
        session_dir=session_dir,
        episode_id=episode_id,
        n_frames=n_frames,
        unique_frames=len(frame_refs),
        net_displacement=displacement,
        duration_ticks=duration_ticks,
        max_displacement=max_displacement,
        yaw_sweep_degrees=yaw_sweep,
        facing_changes=facing_changes,
        unique_facings=len(facings_seen),
        completed=completed,
        termination_reason=termination_reason,
        pixel_sources=pixel_sources,
    )


def validate_recording_quality(
    quality: EpisodeRecordingQuality,
    *,
    name: str = "recording",
    min_blocks_per_tick: float = 0.0,
    min_unique_frame_fraction: float = 0.0,
    max_blocks_per_tick: Optional[float] = None,
    min_yaw_sweep_degrees: float = 0.0,
    min_unique_facings: int = 0,
    require_completed: bool = True,
) -> List[str]:
    """Check one episode's measured quality against a scenario's
    expectations (0/``None`` = no expectation); returns human-readable issue
    strings (empty = healthy). One episode's worth of
    ``validate_recordings``'s per-episode checks, without the
    cross-session pixel-source bookkeeping."""

    where = f"{quality.session_dir}/{quality.episode_id}"
    issues: List[str] = []
    if quality.n_frames == 0:
        return [f"{where}: no pixel frames recorded (record_frames off?)"]
    if min_unique_frame_fraction > 0.0 and quality.unique_frame_fraction < min_unique_frame_fraction:
        issues.append(
            f"{where}: only {quality.unique_frames}/{quality.n_frames} unique pixel "
            f"frames ({quality.unique_frame_fraction:.1%} < "
            f"{min_unique_frame_fraction:.1%}) -- a near-static view has no {name!r} "
            "signal to learn"
        )
    if (
        min_blocks_per_tick > 0.0
        and quality.duration_ticks > 0
        and quality.blocks_per_tick < min_blocks_per_tick
    ):
        issues.append(
            f"{where}: net displacement {quality.net_displacement:.2f} over "
            f"{quality.duration_ticks} ticks ({quality.blocks_per_tick:.4f}/tick < "
            f"{min_blocks_per_tick}/tick) -- the agent barely moved (stuck against "
            "an obstacle?)"
        )
    if (
        max_blocks_per_tick is not None
        and quality.duration_ticks > 0
        and quality.max_blocks_per_tick > max_blocks_per_tick
    ):
        issues.append(
            f"{where}: the agent strayed {quality.max_displacement:.2f} from its "
            f"start ({quality.max_blocks_per_tick:.4f}/tick > {max_blocks_per_tick}/tick) "
            f"-- {name!r} expects a stationary agent (live-server knockback/water/mobs?)"
        )
    if min_yaw_sweep_degrees > 0.0 and quality.yaw_sweep_degrees < min_yaw_sweep_degrees:
        issues.append(
            f"{where}: total yaw sweep {quality.yaw_sweep_degrees:.0f} degrees < "
            f"{min_yaw_sweep_degrees:.0f} -- {name!r} needs the view to actually rotate"
        )
    if min_unique_facings > 0 and quality.unique_facings < min_unique_facings:
        issues.append(
            f"{where}: only {quality.unique_facings} unique facing(s) observed "
            f"(< {min_unique_facings}) -- {name!r} needs the agent to face multiple "
            "directions"
        )
    if require_completed and quality.completed is False:
        issues.append(
            f"{where}: episode terminated early "
            f"({quality.termination_reason or 'unknown reason'}) -- a scripted micro-"
            f"scenario recording that died mid-episode is not the scenario it claims to be"
        )
    return issues


def validate_recordings(
    session_dirs: Sequence[str],
    *,
    name: str = "recording",
    min_blocks_per_tick: float = 0.0,
    min_unique_frame_fraction: float = 0.0,
    max_blocks_per_tick: Optional[float] = None,
    min_yaw_sweep_degrees: float = 0.0,
    min_unique_facings: int = 0,
    require_completed: bool = True,
    expected_pixel_source: Optional[str] = None,
) -> List[str]:
    """Check every recorded episode across ``session_dirs`` against a
    scenario's data-quality expectations, plus the cross-session checks that
    only make sense over a whole training pool: no mixed pixel provenance
    within or across sessions, and (when set) provenance matching
    ``expected_pixel_source``."""

    issues: List[str] = []
    sources_seen: Dict[str, List[str]] = {}
    for session_dir in session_dirs:
        for episode_id in list_episodes(session_dir):
            quality = measure_recording_quality(session_dir, episode_id)
            where = f"{session_dir}/{episode_id}"
            issues += validate_recording_quality(
                quality,
                name=name,
                min_blocks_per_tick=min_blocks_per_tick,
                min_unique_frame_fraction=min_unique_frame_fraction,
                max_blocks_per_tick=max_blocks_per_tick,
                min_yaw_sweep_degrees=min_yaw_sweep_degrees,
                min_unique_facings=min_unique_facings,
                require_completed=require_completed,
            )
            if quality.pixel_sources:
                sources_seen[where] = sorted(set(quality.pixel_sources))
                if len(sources_seen[where]) > 1:
                    issues.append(
                        f"{where}: mixed pixel sources within one episode "
                        f"({sources_seen[where]}) -- the observation distribution changed "
                        "mid-recording (viewer died and fell back to the grid?)"
                    )
                if (
                    expected_pixel_source is not None
                    and sources_seen[where] != [expected_pixel_source]
                ):
                    issues.append(
                        f"{where}: pixel source {sources_seen[where]} != expected "
                        f"{expected_pixel_source!r} -- the requested render path was not "
                        "the one that produced these frames"
                    )

    distinct = {tuple(v) for v in sources_seen.values()}
    if len(distinct) > 1:
        issues.append(
            "sessions mix pixel sources across episodes "
            f"({sorted(sources_seen.items())}) -- do not train one model on frames from "
            "different render paths"
        )
    return issues


@dataclass
class RecordingVerdict:
    """Green/amber/red summary for one session -- the shape the read-only
    clinic backend (phase 8) will read per session."""

    verdict: str  # "green" | "amber" | "red"
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _amber_warnings(
    quality: EpisodeRecordingQuality,
    *,
    min_blocks_per_tick: float,
    min_unique_frame_fraction: float,
) -> List[str]:
    """Soft risk factors on an episode that already clears every hard floor:
    borderline margins, or provenance metadata that predates tracking (so it
    can be neither confirmed nor refuted)."""
    where = f"{quality.session_dir}/{quality.episode_id}"
    warnings: List[str] = []
    if not quality.pixel_sources:
        warnings.append(f"{where}: no pixel provenance recorded (predates provenance tracking)")
    if (
        min_blocks_per_tick > 0.0
        and quality.duration_ticks > 0
        and quality.blocks_per_tick < min_blocks_per_tick * AMBER_MARGIN
    ):
        warnings.append(
            f"{where}: motion {quality.blocks_per_tick:.4f}/tick is within "
            f"{AMBER_MARGIN}x of the {min_blocks_per_tick}/tick floor"
        )
    if (
        min_unique_frame_fraction > 0.0
        and quality.unique_frame_fraction < min_unique_frame_fraction * AMBER_MARGIN
    ):
        warnings.append(
            f"{where}: unique-frame fraction {quality.unique_frame_fraction:.1%} is within "
            f"{AMBER_MARGIN}x of the {min_unique_frame_fraction:.1%} floor"
        )
    return warnings


def verdict_for_session(
    session_dir: str,
    *,
    name: str = "recording",
    min_blocks_per_tick: float = 0.0,
    min_unique_frame_fraction: float = 0.0,
    max_blocks_per_tick: Optional[float] = None,
    min_yaw_sweep_degrees: float = 0.0,
    min_unique_facings: int = 0,
    require_completed: bool = True,
    expected_pixel_source: Optional[str] = None,
) -> RecordingVerdict:
    """Green/amber/red verdict for every episode in one session, combined
    worst-episode-wins (red > amber > green)."""

    issues: List[str] = []
    warnings: List[str] = []
    for episode_id in list_episodes(session_dir):
        quality = measure_recording_quality(session_dir, episode_id)
        issues += validate_recording_quality(
            quality,
            name=name,
            min_blocks_per_tick=min_blocks_per_tick,
            min_unique_frame_fraction=min_unique_frame_fraction,
            max_blocks_per_tick=max_blocks_per_tick,
            min_yaw_sweep_degrees=min_yaw_sweep_degrees,
            min_unique_facings=min_unique_facings,
            require_completed=require_completed,
        )
        if (
            expected_pixel_source is not None
            and quality.pixel_sources
            and sorted(set(quality.pixel_sources)) != [expected_pixel_source]
        ):
            issues.append(
                f"{session_dir}/{episode_id}: pixel source "
                f"{sorted(set(quality.pixel_sources))} != expected "
                f"{expected_pixel_source!r}"
            )
        if not issues:
            warnings += _amber_warnings(
                quality,
                min_blocks_per_tick=min_blocks_per_tick,
                min_unique_frame_fraction=min_unique_frame_fraction,
            )
    verdict = "red" if issues else ("amber" if warnings else "green")
    return RecordingVerdict(verdict=verdict, issues=issues, warnings=warnings)
