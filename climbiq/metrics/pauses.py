"""Pause detection: flag STATIC phases long enough to count as a deliberate rest.

Not every STATIC phase is a "pause" worth surfacing to a climber -- brief
stillness between hand movements is a normal part of climbing, not
something worth flagging. A pause, in the coaching-relevant sense, is
stillness that goes on long enough to represent a deliberate rest, a
route-reading stop, or a shake-out -- not just the natural cadence of
movement.

Responsibility of this module, and only this module: MovementPhase list in,
Pause list out. No phase classification (that's climbiq.metrics.phases),
no smoothness scoring, no report formatting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from climbiq.metrics.phases import MovementPhase, PhaseType


@dataclass(frozen=True, slots=True)
class PauseDetectionConfig:
    """Threshold for what counts as a pause rather than ordinary stillness.

    Attributes:
        min_pause_duration_s: A STATIC phase must last at least this long
            to be reported as a pause. Below this, it's treated as normal
            climbing cadence, not something to flag.

    Like the other threshold defaults in this codebase (SmoothingConfig,
    PhaseSegmentationConfig), this is a starting point, not a validated
    value -- expect to tune it once there's real reference footage with
    an eyeballed or labeled sense of "that was clearly a rest" to compare
    against.
    """

    min_pause_duration_s: float = 1.0

    def __post_init__(self) -> None:
        if self.min_pause_duration_s <= 0:
            raise ValueError(
                f"min_pause_duration_s must be > 0, got {self.min_pause_duration_s!r}"
            )


@dataclass(frozen=True, slots=True)
class Pause:
    """One detected pause -- a STATIC phase that met the duration threshold."""

    start_frame_index: int
    end_frame_index: int
    start_timestamp_s: float
    end_timestamp_s: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class PauseSummary:
    """Aggregate view over a list of detected pauses, for report-level use.

    Attributes:
        pauses: The full list of detected pauses, in chronological order.
        total_pause_time_s: Sum of every pause's duration.
        longest_pause: The single longest pause, or None if there were no
            pauses at all.
    """

    pauses: list[Pause]
    total_pause_time_s: float
    longest_pause: Pause | None


def detect_pauses(
    phases: Sequence[MovementPhase],
    config: PauseDetectionConfig | None = None,
) -> list[Pause]:
    """Extract STATIC phases meeting the pause-duration threshold.

    Args:
        phases: Movement phases, as produced by
            `climbiq.metrics.phases.segment_phases`.
        config: Duration threshold. Uses PauseDetectionConfig()'s default
            if not provided.

    Returns:
        Pauses in chronological order. MOVING and UNKNOWN phases are never
        considered pauses regardless of duration -- a data gap is not
        evidence of a rest, and is left to be understood as a data-quality
        issue (see PhaseType.UNKNOWN), not conflated with one.
    """
    if config is None:
        config = PauseDetectionConfig()

    return [
        Pause(
            start_frame_index=phase.start_frame_index,
            end_frame_index=phase.end_frame_index,
            start_timestamp_s=phase.start_timestamp_s,
            end_timestamp_s=phase.end_timestamp_s,
            duration_s=phase.duration_s,
        )
        for phase in phases
        if phase.phase_type == PhaseType.STATIC and phase.duration_s >= config.min_pause_duration_s
    ]


def summarize_pauses(pauses: Sequence[Pause]) -> PauseSummary:
    """Aggregate a list of pauses into report-ready totals."""
    pauses_list = list(pauses)
    total = sum(p.duration_s for p in pauses_list)
    longest = max(pauses_list, key=lambda p: p.duration_s) if pauses_list else None
    return PauseSummary(pauses=pauses_list, total_pause_time_s=total, longest_pause=longest)