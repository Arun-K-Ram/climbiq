"""Movement-phase segmentation: classify each frame as STATIC or MOVING.

Uses whole-body center-of-mass speed as the signal, not any single limb --
the most robust movement indicator currently available in the pipeline,
and a defensible v1 (see module-level design note below on why the fuller
reach/match/reposition/dyno phase model from the project roadmap is
deliberately deferred).

Responsibility of this module, and only this module: a per-frame speed
signal in, a list of movement-phase segments out. No pause/crux
thresholding on top of these phases, no smoothness/efficiency scoring --
those are sibling metrics modules that consume `MovementPhase` output.

Design note on scope: the original project roadmap envisioned a 4-state
phase model (STATIC_HOLD, REACHING, REPOSITIONING, MATCHING/DYNAMIC).
Distinguishing those reliably needs per-limb signal (which hand/foot is
moving, and how), not just whole-body CoM speed -- that's a real jump in
complexity and a real dependency on the movement-representation work
still ahead. STATIC vs MOVING is the honest, currently-supportable
version: two states the CoM speed signal can actually distinguish, not
four states dressed up to look more sophisticated than the underlying
signal supports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from climbiq.kinematics.derivatives import KinematicSample, magnitude
from climbiq.pose.types import PoseSequence


class PhaseType(Enum):
    """Per-frame or per-segment movement classification.

    STATIC: CoM speed is low -- the climber is holding position.
    MOVING: CoM speed is high enough to count as active movement.
    UNKNOWN: No reliable speed signal for this frame (no CoM estimate,
        or not enough continuous history for a velocity -- see
        `climbiq.kinematics.derivatives`). Deliberately distinct from
        STATIC: a frame with no data is not evidence the climber was
        still, and treating it as such would fabricate a claim the data
        doesn't support. Most commonly seen at the very start of a
        sequence and in the frames immediately following a detection gap.
    """

    STATIC = "static"
    MOVING = "moving"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PhaseSegmentationConfig:
    """Tuning parameters for STATIC/MOVING classification.

    Attributes:
        static_enter_speed: While currently MOVING, speed must drop below
            this threshold (meters/second, assuming a WORLD_METERS
            PoseSequence) to transition into STATIC.
        static_exit_speed: While currently STATIC, speed must rise above
            this threshold to transition into MOVING.
        min_phase_duration_s: Minimum duration for a STATIC or MOVING
            segment to stand on its own. Shorter segments are merged into
            an adjacent segment -- see `segment_phases` for the merge
            policy. Does not apply to UNKNOWN segments, which are
            preserved regardless of duration; see PhaseType.UNKNOWN.

    static_exit_speed must be >= static_enter_speed. Using the same value
    for both disables hysteresis entirely (a single threshold that
    flickers at the boundary); using a genuinely higher exit threshold is
    what prevents a climber holding position right at the edge of the
    threshold from generating dozens of spurious one-frame phase
    transitions from sensor/estimation noise alone.

    These defaults are starting points, not validated against real
    climbing footage -- same caveat as `SmoothingConfig` and
    `DerivativeConfig`. Expect to tune them once there's a labeled or
    eyeballed reference (Phase 0 data collection, per the project
    roadmap) to compare classified phases against.
    """

    static_enter_speed: float = 0.05
    static_exit_speed: float = 0.15
    min_phase_duration_s: float = 0.2

    def __post_init__(self) -> None:
        if self.static_enter_speed <= 0:
            raise ValueError(
                f"static_enter_speed must be > 0, got {self.static_enter_speed!r}"
            )
        if self.static_exit_speed < self.static_enter_speed:
            raise ValueError(
                f"static_exit_speed ({self.static_exit_speed!r}) must be >= "
                f"static_enter_speed ({self.static_enter_speed!r})"
            )
        if self.min_phase_duration_s <= 0:
            raise ValueError(
                f"min_phase_duration_s must be > 0, got {self.min_phase_duration_s!r}"
            )


@dataclass(frozen=True, slots=True)
class MovementPhase:
    """One contiguous segment of frames sharing the same PhaseType.

    Attributes:
        phase_type: STATIC, MOVING, or UNKNOWN for this segment.
        start_frame_index: Index of the first frame in this segment.
        end_frame_index: Index of the last frame in this segment (inclusive).
        start_timestamp_s: Timestamp of the first frame in this segment.
        end_timestamp_s: Timestamp of the last frame in this segment.

    Note on `duration_s`: computed as end_timestamp_s - start_timestamp_s,
    i.e. time from the *start* of the first frame to the *start* of the
    last frame -- not including the last frame's own display duration.
    For a single-frame segment this is exactly 0.0. This slightly
    undercounts true elapsed time (by about one frame period), a
    deliberate simplification to avoid needing to thread fps through
    every call site just for this; not expected to matter at the
    granularity these phases are analyzed at.
    """

    phase_type: PhaseType
    start_frame_index: int
    end_frame_index: int
    start_timestamp_s: float
    end_timestamp_s: float

    @property
    def duration_s(self) -> float:
        return self.end_timestamp_s - self.start_timestamp_s

    @property
    def frame_count(self) -> int:
        return self.end_frame_index - self.start_frame_index + 1


def segment_phases(
    sequence: PoseSequence,
    com_kinematics: Sequence[KinematicSample | None],
    config: PhaseSegmentationConfig | None = None,
) -> list[MovementPhase]:
    """Classify every frame as STATIC/MOVING/UNKNOWN and group into segments.

    Args:
        sequence: The PoseSequence `com_kinematics` was derived from --
            used only for frame_index/timestamp_s, to build segment
            boundaries.
        com_kinematics: Center-of-mass kinematic samples, index-aligned
            with `sequence` (as produced by
            `climbiq.kinematics.derivatives.compute_com_derivatives`).
        config: Classification thresholds and merge policy. Uses
            PhaseSegmentationConfig()'s defaults if not provided.

    Returns:
        A list of MovementPhase segments in frame order, covering every
        frame in `sequence` exactly once (segments are contiguous and
        non-overlapping).

    Raises:
        ValueError: if `sequence` and `com_kinematics` differ in length.
    """
    if config is None:
        config = PhaseSegmentationConfig()
    if len(sequence) != len(com_kinematics):
        raise ValueError(
            f"sequence and com_kinematics must be the same length, got "
            f"{len(sequence)} and {len(com_kinematics)}"
        )
    if len(sequence) == 0:
        return []

    frame_states = _classify_frames(com_kinematics, config)
    raw_segments = _group_into_segments(sequence, frame_states)
    return _merge_short_segments(raw_segments, config.min_phase_duration_s)


def _classify_frames(
    com_kinematics: Sequence[KinematicSample | None],
    config: PhaseSegmentationConfig,
) -> list[PhaseType]:
    """Per-frame hysteresis classification -- see PhaseSegmentationConfig."""
    states: list[PhaseType] = []
    current_state = PhaseType.UNKNOWN

    for sample in com_kinematics:
        if sample is None or sample.velocity is None:
            current_state = PhaseType.UNKNOWN
            states.append(PhaseType.UNKNOWN)
            continue

        speed = magnitude(sample.velocity)
        if current_state == PhaseType.MOVING:
            new_state = (
                PhaseType.STATIC if speed < config.static_enter_speed else PhaseType.MOVING
            )
        elif current_state == PhaseType.STATIC:
            new_state = (
                PhaseType.MOVING if speed > config.static_exit_speed else PhaseType.STATIC
            )
        else:
            # Coming from UNKNOWN, there's no prior state to bias the
            # decision -- classify against the (higher) exit threshold,
            # the same conservative choice as if we'd been STATIC.
            new_state = (
                PhaseType.MOVING if speed >= config.static_exit_speed else PhaseType.STATIC
            )

        states.append(new_state)
        current_state = new_state

    return states


def _group_into_segments(
    sequence: PoseSequence, states: Sequence[PhaseType]
) -> list[MovementPhase]:
    """Group consecutive same-state frames into raw (unmerged) segments."""
    segments: list[MovementPhase] = []
    start_index = 0

    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start_index]:
            start_frame = sequence[start_index]
            end_frame = sequence[i - 1]
            segments.append(
                MovementPhase(
                    phase_type=states[start_index],
                    start_frame_index=start_frame.frame_index,
                    end_frame_index=end_frame.frame_index,
                    start_timestamp_s=start_frame.timestamp_s,
                    end_timestamp_s=end_frame.timestamp_s,
                )
            )
            start_index = i

    return segments


def _merge_short_segments(
    segments: list[MovementPhase], min_duration_s: float
) -> list[MovementPhase]:
    """Absorb STATIC/MOVING segments shorter than min_duration_s into a neighbor.

    UNKNOWN segments are never merged away, regardless of duration -- they
    represent genuine data gaps, not classification noise. A short
    STATIC/MOVING segment adjacent only to UNKNOWN segments (or at a
    sequence boundary with no STATIC/MOVING neighbor at all) is left as
    its own segment rather than being folded into an UNKNOWN region,
    which would discard real speed evidence to satisfy a tidiness
    threshold. Prefers merging into the *next* segment when both
    neighbors are eligible, purely for a stable, deterministic result.
    A final pass coalesces any same-type segments left adjacent to each
    other by a merge (see `_coalesce_adjacent_same_type`).
    """
    segments = list(segments)
    changed = True

    while changed:
        changed = False
        for i, segment in enumerate(segments):
            if segment.phase_type == PhaseType.UNKNOWN:
                continue
            if segment.duration_s >= min_duration_s:
                continue

            target_index: int | None = None
            if i + 1 < len(segments) and segments[i + 1].phase_type != PhaseType.UNKNOWN:
                target_index = i + 1
            elif i - 1 >= 0 and segments[i - 1].phase_type != PhaseType.UNKNOWN:
                target_index = i - 1

            if target_index is None:
                continue

            segments = _merge_two(segments, i, target_index)
            changed = True
            break  # restart the scan; indices shifted after the merge

    # A merge can leave two segments of the same phase_type adjacent to
    # each other (e.g. MOVING, [short STATIC blip merged away], MOVING ->
    # the two MOVING segments are now touching but still separate). Fold
    # any such runs into one segment; this needs only a single final
    # left-to-right pass since coalescing can't create new short segments
    # that would need re-merging above.
    return _coalesce_adjacent_same_type(segments)


def _coalesce_adjacent_same_type(segments: list[MovementPhase]) -> list[MovementPhase]:
    if not segments:
        return []

    coalesced = [segments[0]]
    for segment in segments[1:]:
        previous = coalesced[-1]
        if segment.phase_type == previous.phase_type:
            coalesced[-1] = MovementPhase(
                phase_type=previous.phase_type,
                start_frame_index=previous.start_frame_index,
                end_frame_index=segment.end_frame_index,
                start_timestamp_s=previous.start_timestamp_s,
                end_timestamp_s=segment.end_timestamp_s,
            )
        else:
            coalesced.append(segment)
    return coalesced


def _merge_two(
    segments: list[MovementPhase], first_index: int, second_index: int
) -> list[MovementPhase]:
    """Merge segments[first_index] and segments[second_index] (adjacent,
    in either order) into one segment, adopting the *second* segment's
    phase_type, and return the resulting list."""
    low, high = sorted((first_index, second_index))
    keeper_type = segments[second_index].phase_type

    merged = MovementPhase(
        phase_type=keeper_type,
        start_frame_index=segments[low].start_frame_index,
        end_frame_index=segments[high].end_frame_index,
        start_timestamp_s=segments[low].start_timestamp_s,
        end_timestamp_s=segments[high].end_timestamp_s,
    )
    return segments[:low] + [merged] + segments[high + 1 :]