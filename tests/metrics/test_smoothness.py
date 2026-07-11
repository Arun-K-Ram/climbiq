"""Tests for climbiq.metrics.smoothness."""

from __future__ import annotations

import math

import pytest

from climbiq.kinematics.com import Point3D
from climbiq.kinematics.derivatives import KinematicSample
from climbiq.metrics.phases import MovementPhase, PhaseType
from climbiq.metrics.smoothness import (
    SmoothnessConfig,
    compute_movement_smoothness,
    compute_smoothness_for_moving_phases,
)
from climbiq.pose.types import CoordinateSpace, PoseFrame, PoseSequence, VideoMetadata

FPS = 10.0  # round dt (0.1s) for hand-computable expected values
DT = 1.0 / FPS


def _sequence(n: int) -> PoseSequence:
    frames = [PoseFrame(frame_index=i, timestamp_s=i * DT, keypoints={}) for i in range(n)]
    return PoseSequence(
        frames=frames,
        coordinate_space=CoordinateSpace.WORLD_METERS,
        metadata=VideoMetadata(fps=FPS),
    )


def _sample(velocity: float | None, jerk: float | None) -> KinematicSample:
    return KinematicSample(
        position=Point3D(0.0, 0.0, 0.0),
        velocity=Point3D(velocity, 0.0, 0.0) if velocity is not None else None,
        acceleration=None,
        jerk=Point3D(jerk, 0.0, 0.0) if jerk is not None else None,
    )


def _phase(start: int, end: int, phase_type: PhaseType = PhaseType.MOVING) -> MovementPhase:
    return MovementPhase(
        phase_type=phase_type,
        start_frame_index=start,
        end_frame_index=end,
        start_timestamp_s=start * DT,
        end_timestamp_s=end * DT,
    )


class TestSmoothnessConfig:
    def test_rejects_non_positive_min_duration(self) -> None:
        with pytest.raises(ValueError):
            SmoothnessConfig(min_duration_s=0.0)

    def test_rejects_min_sample_count_below_two(self) -> None:
        with pytest.raises(ValueError):
            SmoothnessConfig(min_sample_count=1)

    def test_accepts_defaults(self) -> None:
        SmoothnessConfig()  # should not raise


class TestComputeMovementSmoothness:
    def test_mismatched_lengths_raise(self) -> None:
        sequence = _sequence(3)
        with pytest.raises(ValueError):
            compute_movement_smoothness(sequence, [None, None], _phase(0, 2))

    def test_hand_verified_ldlj_value(self) -> None:
        """5 frames, constant velocity magnitude 2.0, constant jerk
        magnitude 1.0, fps=10 (dt=0.1s). Phase spans frames 0-4, so
        duration_s = 4*DT - 0*DT = 0.4s (per MovementPhase.duration_s's
        documented start-to-start convention).

        integrated_squared_jerk = 5 samples * 1.0^2 * dt(0.1) = 0.5
        dimensionless_jerk = -(0.4^3 / 2.0^2) * 0.5 = -0.008
        LDLJ = -ln(0.008)
        """
        n = 5
        sequence = _sequence(n)
        kinematics = [_sample(velocity=2.0, jerk=1.0) for _ in range(n)]
        phase = _phase(0, n - 1)

        result = compute_movement_smoothness(sequence, kinematics, phase)

        expected_ldlj = -math.log((0.4**3 / 2.0**2) * 0.5)
        assert result.log_dimensionless_jerk == pytest.approx(expected_ldlj)
        assert result.peak_speed == pytest.approx(2.0)
        assert result.sample_count == 5
        assert result.coverage == pytest.approx(1.0)

    def test_higher_jerk_scores_lower_ldlj(self) -> None:
        """Smoothness should decrease (lower LDLJ) as jerk increases, for
        an otherwise identical movement -- checks the score responds in
        the right *direction*, independent of the exact formula details
        already covered by the hand-verified test above."""
        n = 5
        sequence = _sequence(n)
        smooth_kinematics = [_sample(velocity=2.0, jerk=0.5) for _ in range(n)]
        jerky_kinematics = [_sample(velocity=2.0, jerk=5.0) for _ in range(n)]
        phase = _phase(0, n - 1)

        smooth_score = compute_movement_smoothness(sequence, smooth_kinematics, phase)
        jerky_score = compute_movement_smoothness(sequence, jerky_kinematics, phase)

        assert smooth_score.log_dimensionless_jerk is not None
        assert jerky_score.log_dimensionless_jerk is not None
        assert smooth_score.log_dimensionless_jerk > jerky_score.log_dimensionless_jerk

    def test_no_velocity_samples_produces_none_score(self) -> None:
        n = 5
        sequence = _sequence(n)
        kinematics = [_sample(velocity=None, jerk=1.0) for _ in range(n)]
        result = compute_movement_smoothness(sequence, kinematics, _phase(0, n - 1))
        assert result.log_dimensionless_jerk is None
        assert result.peak_speed is None

    def test_no_jerk_samples_produces_none_score(self) -> None:
        n = 5
        sequence = _sequence(n)
        kinematics = [_sample(velocity=2.0, jerk=None) for _ in range(n)]
        result = compute_movement_smoothness(sequence, kinematics, _phase(0, n - 1))
        assert result.log_dimensionless_jerk is None
        assert result.peak_speed == pytest.approx(2.0)  # velocity data still reported
        assert result.sample_count == 0
        assert result.coverage == pytest.approx(0.0)

    def test_partial_jerk_coverage_is_reported(self) -> None:
        n = 10
        sequence = _sequence(n)
        kinematics = [_sample(velocity=2.0, jerk=1.0 if i >= 5 else None) for i in range(n)]
        result = compute_movement_smoothness(sequence, kinematics, _phase(0, n - 1))
        assert result.sample_count == 5
        assert result.coverage == pytest.approx(0.5)
        # still computes a score from the available samples
        assert result.log_dimensionless_jerk is not None

    def test_all_none_samples_produce_none_score_and_zero_coverage(self) -> None:
        n = 5
        sequence = _sequence(n)
        kinematics: list[KinematicSample | None] = [None] * n
        result = compute_movement_smoothness(sequence, kinematics, _phase(0, n - 1))
        assert result.log_dimensionless_jerk is None
        assert result.peak_speed is None
        assert result.coverage == pytest.approx(0.0)

    def test_short_duration_window_returns_none_despite_valid_data(self) -> None:
        """A 2-frame window (well under the default 0.3s min_duration_s)
        should not produce a score, even with otherwise perfectly valid
        velocity/jerk data -- this is the exact failure mode that
        motivated adding the guard (a spuriously extreme score on a
        near-instantaneous window)."""
        n = 2
        sequence = _sequence(n)
        kinematics = [_sample(velocity=0.8, jerk=1.0) for _ in range(n)]
        result = compute_movement_smoothness(sequence, kinematics, _phase(0, n - 1))
        assert result.log_dimensionless_jerk is None
        # peak_speed/sample_count still reported, so the caller can see
        # *why* the score was withheld rather than just seeing None.
        assert result.peak_speed == pytest.approx(0.8)
        assert result.sample_count == 2

    def test_too_few_samples_returns_none_despite_sufficient_duration(self) -> None:
        """A phase can be long enough in wall-clock time but still have
        too few *valid jerk samples* (e.g. adjacent to a gap) to trust --
        duration and sample count are independent guards."""
        n = 20  # 2.0s at fps=10, well over min_duration_s
        sequence = _sequence(n)
        # Only 2 of 20 frames have a valid jerk value -- under the
        # default min_sample_count of 4.
        kinematics = [_sample(velocity=0.5, jerk=1.0 if i < 2 else None) for i in range(n)]
        result = compute_movement_smoothness(sequence, kinematics, _phase(0, n - 1))
        assert result.log_dimensionless_jerk is None
        assert result.sample_count == 2

    def test_custom_config_thresholds(self) -> None:
        """A window that fails the default guards should succeed once a
        more permissive config is supplied explicitly."""
        n = 2
        sequence = _sequence(n)
        kinematics = [_sample(velocity=0.8, jerk=1.0) for _ in range(n)]
        permissive_config = SmoothnessConfig(min_duration_s=0.01, min_sample_count=2)
        result = compute_movement_smoothness(
            sequence, kinematics, _phase(0, n - 1), permissive_config
        )
        assert result.log_dimensionless_jerk is not None


class TestComputeSmoothnessForMovingPhases:
    def test_only_scores_moving_phases(self) -> None:
        n = 15
        sequence = _sequence(n)
        kinematics = [_sample(velocity=2.0, jerk=1.0) for _ in range(n)]
        phases = [
            _phase(0, 4, PhaseType.STATIC),
            _phase(5, 9, PhaseType.MOVING),
            _phase(10, 14, PhaseType.UNKNOWN),
        ]
        result = compute_smoothness_for_moving_phases(sequence, kinematics, phases)
        assert len(result) == 1
        assert result[0].start_frame_index == 5
        assert result[0].end_frame_index == 9

    def test_empty_phase_list_returns_empty(self) -> None:
        sequence = _sequence(5)
        kinematics = [_sample(velocity=2.0, jerk=1.0) for _ in range(5)]
        assert compute_smoothness_for_moving_phases(sequence, kinematics, []) == []