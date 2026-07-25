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
from typing import Any, Dict, List, Optional, Sequence



def list_episodes(session_dir: str) -> List[str]:
    """List episode ids directly from the Record without loading a runtime."""

    suffix = ".streams.jsonl"
    return sorted(
        name[: -len(suffix)]
        for name in os.listdir(session_dir)
        if name.startswith("episode_") and name.endswith(suffix)
    )

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
    #: Semantic-grid signals (issue #202, ``vision.frame.grid`` -- a world's
    #: egocentric material/object crop, where the world publishes one; 0 for
    #: recordings/worlds without it). These exist because pixel-hash
    #: uniqueness alone cannot tell genuine world motion from a static scene
    #: whose pixels merely animate (HUD flicker, dithering): a scene can be
    #: 100% unique pixel frames while the semantic grid -- and the agent's
    #: position -- never change.
    semantic_frames: int = 0
    unique_semantic_frames: int = 0
    #: Fraction of frame-to-frame transitions where the semantic grid
    #: actually changed (both frames' grids known).
    semantic_change_fraction: float = 0.0
    #: Fraction of frame-to-frame transitions where the agent's
    #: forward-filled grid position actually changed -- the real ego-motion
    #: signal ``walk_forward_short``/``blocked_forward`` gate on, since a
    #: rotating view or an animated HUD can vary pixels/semantics without
    #: the agent moving at all.
    moving_transition_fraction: float = 0.0
    #: Longest run of consecutive "position unchanged" transitions ending at
    #: the very last frame -- an *accidental* long stationary tail (the
    #: agent walked into an obstacle/boundary and idled for the rest of the
    #: episode) vs. an intentional, explicitly bounded blocked phase
    #: (``blocked_forward``) both show up as position-unchanged runs, but
    #: only a long *trailing* one is the regression this catches.
    longest_stationary_tail: int = 0
    #: Longest run of consecutive "position unchanged" transitions anywhere
    #: in the episode (including a deliberate mid-episode blocked phase).
    longest_stationary_run: int = 0
    #: Count of frame-to-frame transitions where the forward-filled position
    #: changed -- the raw numerator behind ``moving_transition_fraction``.
    position_change_count: int = 0

    @property
    def unique_frame_fraction(self) -> float:
        return self.unique_frames / self.n_frames if self.n_frames else 0.0

    @property
    def blocks_per_tick(self) -> float:
        return self.net_displacement / self.duration_ticks if self.duration_ticks else 0.0

    @property
    def max_blocks_per_tick(self) -> float:
        return self.max_displacement / self.duration_ticks if self.duration_ticks else 0.0

    @property
    def unique_semantic_frame_fraction(self) -> float:
        return self.unique_semantic_frames / self.semantic_frames if self.semantic_frames else 0.0


def _wrapped_degrees(delta: float) -> float:
    return abs((delta + 180.0) % 360.0 - 180.0)


def _grid_key(grid: Any) -> Optional[tuple]:
    """Hashable snapshot of a semantic-grid payload (a list of lists of
    ints), or ``None`` for anything else."""
    if not isinstance(grid, list):
        return None
    return tuple(tuple(row) for row in grid)


def _longest_run(flags: Sequence[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _trailing_run(flags: Sequence[bool]) -> int:
    count = 0
    for flag in reversed(flags):
        if not flag:
            break
        count += 1
    return count


def measure_recording_quality(session_dir: str, episode_id: str) -> EpisodeRecordingQuality:
    """Scan one episode's stream log for the gate's signals: unique pixel
    frames (via content-hash ``frame_ref``), x/y displacement (net and max),
    yaw sweep and/or discrete facing changes, episode completion, pixel
    provenance, and semantic-grid/forward-filled-position motion signals
    (issue #202). World-agnostic: reads whatever subset of
    ``spatial.position``/``spatial.rotation``/``spatial.facing``/
    ``vision.frame.grid`` the recording's world actually publishes.

    Semantic/position signals are read per cognitive tick, not per log line:
    ``spatial.position``/``vision.frame.grid`` are delta-published (a line
    appears only when the value changes), while ``vision.frame.pixels`` is
    republished every visible tick, so each pixel frame's position/grid must
    be forward-filled from the most recent update *as of that same tick* --
    every line one tick publishes shares that tick's ``timestamp``.
    """

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

    position_at_frame: List[Optional[tuple]] = []
    grid_key_at_frame: List[Optional[tuple]] = []
    known_position: Optional[tuple] = None
    known_grid_key: Optional[tuple] = None
    semantic_grids_seen: set = set()

    def flush_tick(records: List[Dict[str, Any]]) -> None:
        nonlocal known_position, known_grid_key, n_frames, first_pos, last_pos, max_displacement
        nonlocal last_yaw, yaw_sweep, last_facing, facing_changes
        has_pixel = False
        for record in records:
            stream_id = record.get("stream_id")
            if stream_id == "vision.frame.pixels":
                has_pixel = True
                n_frames += 1
                ref = record.get("frame_ref") or record.get("hash")
                if ref:
                    frame_refs.add(ref)
            elif stream_id == "vision.frame.grid":
                grid_key = _grid_key(record.get("payload"))
                if grid_key is not None:
                    known_grid_key = grid_key
                    semantic_grids_seen.add(grid_key)
            elif stream_id == "spatial.position":
                payload = record.get("payload") or {}
                # Minecraft's position is 3-D ({x, y, z}, y = height) -- the
                # horizontal plane the gate cares about is x/z. Crafter's is
                # a genuine 2-D grid ({x, y}, no z key), so fall back to y
                # only when z is absent.
                horizontal = payload.get("z", payload.get("y", 0.0))
                pos = (float(payload.get("x", 0.0)), float(horizontal))
                known_position = pos
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
        if has_pixel:
            position_at_frame.append(known_position)
            grid_key_at_frame.append(known_grid_key)

    streams_path = os.path.join(session_dir, f"{episode_id}.streams.jsonl")
    tick_timestamp: Any = object()  # sentinel: never equals a real timestamp
    tick_records: List[Dict[str, Any]] = []
    with open(streams_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            timestamp = record.get("timestamp")
            if tick_records and timestamp != tick_timestamp:
                flush_tick(tick_records)
                tick_records = []
            tick_timestamp = timestamp
            tick_records.append(record)
    if tick_records:
        flush_tick(tick_records)

    displacement = (
        math.hypot(last_pos[0] - first_pos[0], last_pos[1] - first_pos[1])
        if first_pos is not None and last_pos is not None
        else 0.0
    )

    semantic_frames = sum(1 for key in grid_key_at_frame if key is not None)
    semantic_changed = 0
    semantic_known_transitions = 0
    stationary_flags: List[bool] = []
    position_changed = 0
    position_known_transitions = 0
    for i in range(len(position_at_frame) - 1):
        pos_a, pos_b = position_at_frame[i], position_at_frame[i + 1]
        moved: Optional[bool] = None
        if pos_a is not None and pos_b is not None:
            moved = pos_a != pos_b
            position_known_transitions += 1
            if moved:
                position_changed += 1
        stationary_flags.append(moved is False)

        grid_a, grid_b = grid_key_at_frame[i], grid_key_at_frame[i + 1]
        if grid_a is not None and grid_b is not None:
            semantic_known_transitions += 1
            if grid_a != grid_b:
                semantic_changed += 1

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
        semantic_frames=semantic_frames,
        unique_semantic_frames=len(semantic_grids_seen),
        semantic_change_fraction=(
            semantic_changed / semantic_known_transitions if semantic_known_transitions else 0.0
        ),
        moving_transition_fraction=(
            position_changed / position_known_transitions if position_known_transitions else 0.0
        ),
        longest_stationary_tail=_trailing_run(stationary_flags),
        longest_stationary_run=_longest_run(stationary_flags),
        position_change_count=position_changed,
    )


def validate_recording_quality(
    quality: EpisodeRecordingQuality,
    *,
    name: str = "recording",
    min_blocks_per_tick: float = 0.0,
    min_unique_frame_fraction: float = 0.0,
    min_unique_frames: int = 0,
    max_blocks_per_tick: Optional[float] = None,
    min_yaw_sweep_degrees: float = 0.0,
    min_unique_facings: int = 0,
    require_completed: bool = True,
    min_moving_transition_fraction: float = 0.0,
    min_semantic_change_fraction: float = 0.0,
    max_longest_stationary_tail: Optional[int] = None,
    max_longest_stationary_run: Optional[int] = None,
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
    if min_unique_frames > 0 and quality.unique_frames < min_unique_frames:
        issues.append(
            f"{where}: only {quality.unique_frames} unique pixel frame(s) "
            f"(< {min_unique_frames}) -- the recording appears frozen"
        )
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
    if (
        min_moving_transition_fraction > 0.0
        and quality.moving_transition_fraction < min_moving_transition_fraction
    ):
        issues.append(
            f"{where}: only {quality.position_change_count} of "
            f"{max(quality.n_frames - 1, 0)} frame transitions actually moved the agent "
            f"({quality.moving_transition_fraction:.1%} < "
            f"{min_moving_transition_fraction:.1%}) -- {name!r} needs sustained ego-motion, "
            "not just varying pixels"
        )
    if (
        min_semantic_change_fraction > 0.0
        and quality.semantic_change_fraction < min_semantic_change_fraction
    ):
        issues.append(
            f"{where}: the semantic grid changed on only "
            f"{quality.semantic_change_fraction:.1%} of transitions "
            f"(< {min_semantic_change_fraction:.1%}) -- pixel-frame uniqueness alone is not "
            f"proof the world moved (a scene can be 100% unique pixels from HUD animation "
            f"or render noise while the scene itself is stationary)"
        )
    if (
        max_longest_stationary_tail is not None
        and quality.longest_stationary_tail > max_longest_stationary_tail
    ):
        issues.append(
            f"{where}: the episode ends with {quality.longest_stationary_tail} consecutive "
            f"transitions with no position change (> {max_longest_stationary_tail}) -- an "
            f"accidental long stationary tail (agent stuck against an obstacle/boundary for "
            f"the rest of the episode), not {name!r}'s explicitly bounded blocked phase"
        )
    if (
        max_longest_stationary_run is not None
        and quality.longest_stationary_run > max_longest_stationary_run
    ):
        issues.append(
            f"{where}: the longest run with no position change is "
            f"{quality.longest_stationary_run} transitions (> {max_longest_stationary_run}) "
            f"-- {name!r}'s blocked phase should be a short, explicitly bounded hold, not an "
            "open-ended stall"
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
    min_moving_transition_fraction: float = 0.0,
    min_semantic_change_fraction: float = 0.0,
    max_longest_stationary_tail: Optional[int] = None,
    max_longest_stationary_run: Optional[int] = None,
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
                min_moving_transition_fraction=min_moving_transition_fraction,
                min_semantic_change_fraction=min_semantic_change_fraction,
                max_longest_stationary_tail=max_longest_stationary_tail,
                max_longest_stationary_run=max_longest_stationary_run,
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

    def as_dict(self) -> Dict[str, object]:
        """Return the stable, JSON-safe contract consumed by the clinic."""

        return {
            "verdict": self.verdict,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


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
    min_unique_frames: int = 0,
    max_blocks_per_tick: Optional[float] = None,
    min_yaw_sweep_degrees: float = 0.0,
    min_unique_facings: int = 0,
    require_completed: bool = True,
    expected_pixel_source: Optional[str] = None,
    min_moving_transition_fraction: float = 0.0,
    min_semantic_change_fraction: float = 0.0,
    max_longest_stationary_tail: Optional[int] = None,
    max_longest_stationary_run: Optional[int] = None,
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
            min_unique_frames=min_unique_frames,
            max_blocks_per_tick=max_blocks_per_tick,
            min_yaw_sweep_degrees=min_yaw_sweep_degrees,
            min_unique_facings=min_unique_facings,
            require_completed=require_completed,
            min_moving_transition_fraction=min_moving_transition_fraction,
            min_semantic_change_fraction=min_semantic_change_fraction,
            max_longest_stationary_tail=max_longest_stationary_tail,
            max_longest_stationary_run=max_longest_stationary_run,
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


def evaluate_record_quality(
    session_dir: str,
    *,
    name: str = "recording",
    min_blocks_per_tick: float = 0.0,
    min_unique_frame_fraction: float = 0.0,
    min_unique_frames: int = 0,
    max_blocks_per_tick: Optional[float] = None,
    min_yaw_sweep_degrees: float = 0.0,
    min_unique_facings: int = 0,
    require_completed: bool = True,
    expected_pixel_source: Optional[str] = None,
    min_moving_transition_fraction: float = 0.0,
    min_semantic_change_fraction: float = 0.0,
    max_longest_stationary_tail: Optional[int] = None,
    max_longest_stationary_run: Optional[int] = None,
) -> RecordingVerdict:
    """Convenience alias for :func:`verdict_for_session`."""
    return verdict_for_session(
        session_dir,
        name=name,
        min_blocks_per_tick=min_blocks_per_tick,
        min_unique_frame_fraction=min_unique_frame_fraction,
        min_unique_frames=min_unique_frames,
        max_blocks_per_tick=max_blocks_per_tick,
        min_yaw_sweep_degrees=min_yaw_sweep_degrees,
        min_unique_facings=min_unique_facings,
        require_completed=require_completed,
        expected_pixel_source=expected_pixel_source,
        min_moving_transition_fraction=min_moving_transition_fraction,
        min_semantic_change_fraction=min_semantic_change_fraction,
        max_longest_stationary_tail=max_longest_stationary_tail,
        max_longest_stationary_run=max_longest_stationary_run,
    )


# --------------------------------------------------------------------------- train/holdout split-overlap audit (issue #202)
#
# A recorded holdout episode is only evidence of generalization if it is
# genuinely different from every training episode -- exact or near-exact
# duplication (the same frames, or almost all of them, in the same order)
# means the "held-out" evaluation is silently re-testing memorized training
# data. This audit reads the same stream logs ``measure_recording_quality``
# does, but compares *across* episodes rather than summarizing one.


@dataclass
class EpisodeFrameSignature:
    """One episode's ordered pixel-frame and semantic-grid identities, for
    cross-episode overlap comparison."""

    session_dir: str
    episode_id: str
    pixel_frame_refs: List[str] = field(default_factory=list)
    semantic_grid_keys: List[Optional[tuple]] = field(default_factory=list)


def episode_frame_signature(session_dir: str, episode_id: str) -> EpisodeFrameSignature:
    """Read one episode's ordered pixel content-hashes (``frame_ref``) and
    semantic-grid keys straight from its stream log, aligned 1:1 by pixel
    frame like ``measure_recording_quality``'s forward-filled signals."""
    pixel_frame_refs: List[str] = []
    semantic_grid_keys: List[Optional[tuple]] = []
    known_grid_key: Optional[tuple] = None
    streams_path = os.path.join(session_dir, f"{episode_id}.streams.jsonl")
    with open(streams_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stream_id = record.get("stream_id")
            if stream_id == "vision.frame.grid":
                grid_key = _grid_key(record.get("payload"))
                if grid_key is not None:
                    known_grid_key = grid_key
            elif stream_id == "vision.frame.pixels":
                ref = record.get("frame_ref") or record.get("hash")
                pixel_frame_refs.append(str(ref) if ref else "")
                semantic_grid_keys.append(known_grid_key)
    return EpisodeFrameSignature(
        session_dir=session_dir,
        episode_id=episode_id,
        pixel_frame_refs=pixel_frame_refs,
        semantic_grid_keys=semantic_grid_keys,
    )


def _corresponding_frame_fraction(a: EpisodeFrameSignature, b: EpisodeFrameSignature) -> float:
    """Fraction of index-aligned pixel frames that match between two
    episodes, over the shorter episode's length (0.0 if either is empty)."""
    n = min(len(a.pixel_frame_refs), len(b.pixel_frame_refs))
    if n == 0:
        return 0.0
    matches = sum(
        1 for i in range(n) if a.pixel_frame_refs[i] and a.pixel_frame_refs[i] == b.pixel_frame_refs[i]
    )
    return matches / n


@dataclass
class SplitOverlapReport:
    """Cross-session train/holdout leakage audit result."""

    #: Pixel frame content-hashes present in both the train and holdout
    #: pools (exact duplicate frames, wherever in their episode they fall).
    exact_pixel_frame_overlap: int = 0
    #: Same, for semantic-grid content.
    exact_semantic_grid_overlap: int = 0
    #: Highest index-aligned matching-frame fraction over every
    #: (train episode, holdout episode) pair.
    max_corresponding_frame_fraction: float = 0.0
    #: (train "session_dir/episode_id", holdout "session_dir/episode_id")
    #: pairs whose corresponding-frame fraction is 1.0 (byte-for-byte
    #: identical recordings, allowing for one being a length-truncated
    #: prefix/suffix of the other).
    duplicate_episode_pairs: List[tuple] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


def audit_split_overlap(
    train_session_dirs: Sequence[str],
    holdout_session_dirs: Sequence[str],
    *,
    max_corresponding_frame_fraction: float = 0.35,
) -> SplitOverlapReport:
    """Compare every recorded train episode against every recorded holdout
    episode and report leakage signals, hard-failing (via ``issues``) when:

    - a holdout episode exactly matches a training episode
      (``duplicate_episode_pairs``, regardless of ``max_corresponding_frame_fraction``);
    - any pair's corresponding-frame overlap exceeds
      ``max_corresponding_frame_fraction``.

    The default threshold is looser than the acceptance target (<1% exact
    corresponding-frame overlap) some scenarios should ultimately hit,
    because a handful of scripted micro-scenarios (``approach_entity``,
    ``object_permanence``) deliberately clear terrain to a flat, featureless
    background before the scripted entity/wall ever enters view -- their
    early frames are near-identical across every episode almost regardless
    of seed, independent of world size, which is expected given the design
    and not itself evidence of a leaked recording. The hard, unconditional
    check -- an exact whole-episode duplicate -- is what actually catches a
    repeated seed or an accidentally un-varied scenario parameter; tighten
    ``max_corresponding_frame_fraction`` per call for scenarios whose design
    doesn't share a background this way.

    A caller that wants to explicitly opt into an in-distribution duplicate
    (e.g. a deliberately repeated smoke-test episode) should filter
    ``session_dirs`` before calling this, or raise the threshold -- this
    function has no opt-out of its own.
    """
    train_signatures = [
        episode_frame_signature(session_dir, episode_id)
        for session_dir in train_session_dirs
        for episode_id in list_episodes(session_dir)
    ]
    holdout_signatures = [
        episode_frame_signature(session_dir, episode_id)
        for session_dir in holdout_session_dirs
        for episode_id in list_episodes(session_dir)
    ]

    train_pixel_refs = {ref for sig in train_signatures for ref in sig.pixel_frame_refs if ref}
    holdout_pixel_refs = {ref for sig in holdout_signatures for ref in sig.pixel_frame_refs if ref}
    train_grid_keys = {
        key for sig in train_signatures for key in sig.semantic_grid_keys if key is not None
    }
    holdout_grid_keys = {
        key for sig in holdout_signatures for key in sig.semantic_grid_keys if key is not None
    }

    report = SplitOverlapReport(
        exact_pixel_frame_overlap=len(train_pixel_refs & holdout_pixel_refs),
        exact_semantic_grid_overlap=len(train_grid_keys & holdout_grid_keys),
    )

    for train_sig in train_signatures:
        train_where = f"{train_sig.session_dir}/{train_sig.episode_id}"
        for holdout_sig in holdout_signatures:
            fraction = _corresponding_frame_fraction(train_sig, holdout_sig)
            report.max_corresponding_frame_fraction = max(
                report.max_corresponding_frame_fraction, fraction
            )
            holdout_where = f"{holdout_sig.session_dir}/{holdout_sig.episode_id}"
            if fraction >= 1.0:
                report.duplicate_episode_pairs.append((train_where, holdout_where))
                report.issues.append(
                    f"holdout episode {holdout_where} exactly matches training episode "
                    f"{train_where} -- this \"held-out\" evaluation would silently "
                    "re-test memorized training data"
                )
            elif fraction > max_corresponding_frame_fraction:
                report.issues.append(
                    f"holdout episode {holdout_where} shares {fraction:.1%} of its "
                    f"corresponding frames with training episode {train_where} "
                    f"(> {max_corresponding_frame_fraction:.1%}) -- too close to a "
                    "duplicate to count as independent evaluation data"
                )
    return report
