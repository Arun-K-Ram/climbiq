"""Movement smoothness scoring via Log Dimensionless Jerk (LDLJ-V).

LDLJ-V is a real, validated metric from the motor-control literature, not
an invented heuristic: Hogan, N. & Sternad, D. (2009), "Sensitivity of
Smoothness Measures to Movement Duration, Amplitude, and Arrests", Journal
of Motor Behavior, established that jerk-based smoothness measures need to
be dimensionless to give meaningful, comparable results across movements
of different speed and duration; Balasubramanian et al. (2012, 2015)
formalized LDLJ-V specifically and showed it (alongside the frequency-
domain SPARC metric) to be one of the few smoothness measures that behaves
consistently under changes in movement duration and amplitude. It has
since been applied directly to whole-body center-of-mass trajectories
(e.g. gait analysis studies comparing CoM-derived LDLJ-V against other
smoothness measures) -- the same kind of signal ClimbIQ computes in
`climbiq.kinematics.com`, which is why this is the metric used here rather
than a simpler but less-validated summary of raw jerk magnitude.

Formula (velocity-based dimensionless jerk, then log-transformed):

    DLJ  = -(T^3 / v_peak^2) * integral[t1,t2] of jerk(t)^2 dt
    LDLJ = -ln(-DLJ) = -ln( (T^3 / v_peak^2) * integral of jerk(t)^2 dt )

where T is the movement's duration, v_peak is its peak speed, and jerk(t)
is the CoM's jerk over the movement. Higher (less negative / more
positive) LDLJ means smoother movement; lower (more negative) means
jerkier. The T^3/v_peak^2 normalization is what makes the measure
dimensionless -- without it, a slow, long movement and a fast, short one
would score very differently purely due to duration/amplitude, not actual
smoothness (this is precisely the failure mode Hogan & Sternad's paper
demonstrated in raw jerk-based measures).

Important honesty about what this score is and isn't: it's computed from
whole-body CoM motion, not any individual limb, so it characterizes
overall body-movement smoothness during a phase, not e.g. "was the left
hand's placement smooth." It's also a relative/comparative measure --
useful for tracking whether a climber's movement on a given move type
gets smoother over sessions, or for flagging the jerkiest move in a climb
relative to their own other moves -- not an absolute pass/fail threshold
with a universally "good" cutoff. No such universal cutoff is established
in the literature for climbing specifically; treat any interpretation
threshold as something to calibrate empirically later, not something to
assume now.

Responsibility of this module, and only this module: CoM kinematics plus a
MOVING phase in, a smoothness score out. No phase classification, no pause
detection, no report formatting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from climbiq.kinematics.derivatives import KinematicSample, magnitude
from climbiq.metrics.phases import MovementPhase, PhaseType
from climbiq.pose.types import PoseSequence


@dataclass(frozen=True, slots=True)
class SmoothnessConfig:
    """Minimum window size for a smoothness score to be treated as meaningful.

    LDLJ-V is sensitive to very short windows: with only a couple of
    samples and a tiny duration T, the T^3 normalization term can swing
    the score to an extreme value (in either direction) that reflects the
    window's brevity rather than the movement's actual smoothness -- a
    real, observed failure mode, not a hypothetical one (a 2-3-frame
    MOVING phase scoring as the single smoothest movement in an entire
    climb, wildly apart from every other score, was the concrete case
    that motivated adding this guard). Below either threshold,
    `log_dimensionless_jerk` is reported as None rather than as a number
    that looks precise but isn't trustworthy.

    Attributes:
        min_duration_s: Minimum phase duration for a score to be computed.
        min_sample_count: Minimum number of valid jerk samples for a score
            to be computed, independent of nominal duration (a phase can
            be long in wall-clock time but still have few usable jerk
            samples if it's adjacent to a detection gap).

    Starting-point defaults, not empirically validated -- same caveat as
    every other threshold config in this codebase.
    """

    min_duration_s: float = 0.3
    min_sample_count: int = 4

    def __post_init__(self) -> None:
        if self.min_duration_s <= 0:
            raise ValueError(f"min_duration_s must be > 0, got {self.min_duration_s!r}")
        if self.min_sample_count < 2:
            raise ValueError(
                f"min_sample_count must be >= 2, got {self.min_sample_count!r} "
                "(LDLJ-V needs at least 2 jerk samples to integrate over)"
            )


@dataclass(frozen=True, slots=True)
class SmoothnessScore:
    """LDLJ-V smoothness result for one movement phase.

    Attributes:
        start_frame_index: First frame of the scored phase.
        end_frame_index: Last frame of the scored phase (inclusive).
        start_timestamp_s: Timestamp of the first frame.
        end_timestamp_s: Timestamp of the last frame.
        log_dimensionless_jerk: The LDLJ-V score, or None if it couldn't
            be computed -- see `compute_movement_smoothness` for the
            specific conditions that produce None (no valid jerk samples
            in the window, zero peak speed, zero/negative duration, a
            degenerate zero-jerk window, or a window too short to trust
            -- see `SmoothnessConfig`). Higher is smoother.
        peak_speed: Maximum CoM speed observed in the window, or None if
            no velocity samples were available. Reported alongside the
            score since it's both an interesting figure on its own and
            the normalization factor the score depends on.
        sample_count: Number of frames in the window that had a valid
            jerk value and contributed to the integral.
        coverage: sample_count divided by the window's total frame count
            -- how much of this phase's jerk signal was actually
            available versus missing (e.g. the first few frames after a
            gap, before jerk's cascading history catches up -- see
            `climbiq.kinematics.derivatives.KinematicSample`). A score
            computed from low coverage should be trusted less than one
            from high coverage; this field makes that explicit rather
            than hiding it, the same philosophy as
            `CenterOfMassEstimate.mass_coverage`.
    """

    start_frame_index: int
    end_frame_index: int
    start_timestamp_s: float
    end_timestamp_s: float
    log_dimensionless_jerk: float | None
    peak_speed: float | None
    sample_count: int
    coverage: float


def compute_movement_smoothness(
    sequence: PoseSequence,
    com_kinematics: Sequence[KinematicSample | None],
    phase: MovementPhase,
    config: SmoothnessConfig | None = None,
) -> SmoothnessScore:
    """Compute LDLJ-V smoothness for the CoM trajectory within one phase.

    Args:
        sequence: The PoseSequence `com_kinematics` was derived from --
            used only for `fps`, needed to numerically integrate jerk
            over time from discrete per-frame samples.
        com_kinematics: Center-of-mass kinematic samples, index-aligned
            with `sequence` (as produced by
            `climbiq.kinematics.derivatives.compute_com_derivatives`).
        phase: The phase to score. Any PhaseType is accepted here (this
            function doesn't enforce MOVING-only), but scoring a STATIC
            or UNKNOWN phase is rarely meaningful -- see
            `compute_smoothness_for_moving_phases` for the intended
            typical entry point, which filters to MOVING phases only.
        config: Minimum-window guards. Uses SmoothnessConfig()'s defaults
            if not provided.

    Returns:
        A SmoothnessScore. `log_dimensionless_jerk` is None if no
        meaningful score could be computed: no valid jerk samples in the
        phase's frame range, no valid velocity samples (so no peak
        speed), a peak speed of zero, a phase duration of zero, an
        (effectively impossible with real, noisy data, but guarded
        anyway) exactly-zero integrated squared jerk, or a window shorter
        than `SmoothnessConfig`'s minimums. `peak_speed`, `sample_count`,
        and `coverage` are still reported even when the score itself is
        None, so a caller can tell *why* a window was excluded rather
        than just seeing an absent value.

    Raises:
        ValueError: if `sequence` and `com_kinematics` differ in length.
    """
    if config is None:
        config = SmoothnessConfig()
    if len(sequence) != len(com_kinematics):
        raise ValueError(
            f"sequence and com_kinematics must be the same length, got "
            f"{len(sequence)} and {len(com_kinematics)}"
        )

    window = com_kinematics[phase.start_frame_index : phase.end_frame_index + 1]
    frame_dt = 1.0 / sequence.fps

    speeds = [
        magnitude(sample.velocity)
        for sample in window
        if sample is not None and sample.velocity is not None
    ]
    peak_speed = max(speeds) if speeds else None

    jerk_squared_values = [
        magnitude(sample.jerk) ** 2
        for sample in window
        if sample is not None and sample.jerk is not None
    ]
    sample_count = len(jerk_squared_values)
    coverage = sample_count / len(window) if window else 0.0

    duration_s = phase.duration_s

    log_dimensionless_jerk: float | None = None
    if duration_s >= config.min_duration_s and sample_count >= config.min_sample_count:
        log_dimensionless_jerk = _try_compute_ldlj(
            jerk_squared_values=jerk_squared_values,
            frame_dt=frame_dt,
            duration_s=duration_s,
            peak_speed=peak_speed,
        )

    return SmoothnessScore(
        start_frame_index=phase.start_frame_index,
        end_frame_index=phase.end_frame_index,
        start_timestamp_s=phase.start_timestamp_s,
        end_timestamp_s=phase.end_timestamp_s,
        log_dimensionless_jerk=log_dimensionless_jerk,
        peak_speed=peak_speed,
        sample_count=sample_count,
        coverage=coverage,
    )


def compute_smoothness_for_moving_phases(
    sequence: PoseSequence,
    com_kinematics: Sequence[KinematicSample | None],
    phases: Sequence[MovementPhase],
    config: SmoothnessConfig | None = None,
) -> list[SmoothnessScore]:
    """Score every MOVING phase in a phase list; skip STATIC and UNKNOWN.

    STATIC phases are skipped because near-zero jerk during intentional
    stillness would trivially score as "very smooth" in a way that's
    meaningless -- smoothness is a property of a movement, and a STATIC
    phase isn't one. UNKNOWN phases are skipped because there's no
    reliable jerk signal to score in the first place.

    This is the intended typical entry point into this module; call
    `compute_movement_smoothness` directly only if you specifically need
    to score a non-MOVING phase for some other reason.
    """
    return [
        compute_movement_smoothness(sequence, com_kinematics, phase, config)
        for phase in phases
        if phase.phase_type == PhaseType.MOVING
    ]


def _try_compute_ldlj(
    jerk_squared_values: Sequence[float],
    frame_dt: float,
    duration_s: float,
    peak_speed: float | None,
) -> float | None:
    if not jerk_squared_values or peak_speed is None:
        return None
    if peak_speed <= 0 or duration_s <= 0:
        return None

    integrated_squared_jerk = sum(jerk_squared_values) * frame_dt
    if integrated_squared_jerk <= 0:
        # Degenerate case: exactly zero jerk throughout. Real, noisy data
        # essentially never produces this exactly; guarded rather than
        # letting math.log(0) raise, since "perfectly smooth" isn't a
        # score this measure is designed to express numerically (the
        # true LDLJ value would be +infinity).
        return None

    dimensionless_jerk = -(duration_s**3 / peak_speed**2) * integrated_squared_jerk
    return -math.log(-dimensionless_jerk)