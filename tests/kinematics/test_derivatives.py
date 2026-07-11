"""Tests for climbiq.kinematics.derivatives.

Uses synthetic motion with exactly known derivatives (constant velocity,
constant acceleration) so results can be checked against hand-computed
expected values, not just "some plausible number came out."
"""

from __future__ import annotations

import pytest

from climbiq.kinematics.com import CenterOfMassEstimate, Point3D
from climbiq.kinematics.derivatives import (
    DerivativeConfig,
    compute_com_derivatives,
    compute_keypoint_derivatives,
    compute_point_derivatives,
    magnitude,
)
from climbiq.pose.types import (
    CoordinateSpace,
    KeypointName,
    PoseFrame,
    PoseKeypoint,
    PoseSequence,
    VideoMetadata,
)

FPS = 30.0
DT = 1.0 / FPS


class TestMagnitude:
    def test_3_4_5_triangle(self) -> None:
        assert magnitude(Point3D(3.0, 4.0, 0.0)) == pytest.approx(5.0)

    def test_missing_z_treated_as_2d(self) -> None:
        assert magnitude(Point3D(3.0, 4.0, None)) == pytest.approx(5.0)

    def test_zero_vector(self) -> None:
        assert magnitude(Point3D(0.0, 0.0, 0.0)) == pytest.approx(0.0)


class TestDerivativeConfig:
    def test_rejects_non_positive_max_gap(self) -> None:
        with pytest.raises(ValueError):
            DerivativeConfig(max_gap_s=0.0)


class TestComputePointDerivatives:
    def test_empty_input_returns_empty(self) -> None:
        assert compute_point_derivatives([], []) == []

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            compute_point_derivatives([0.0, 1.0], [Point3D(0, 0, 0)])

    def test_first_sample_has_no_velocity(self) -> None:
        points = [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)]
        timestamps = [0.0, 1.0]
        result = compute_point_derivatives(timestamps, points)
        assert result[0] is not None
        assert result[0].velocity is None

    def test_constant_velocity_motion(self) -> None:
        """x(t) = 2t, dt=1 -> velocity should be exactly 2.0 from sample 1
        onward, acceleration exactly 0.0 from sample 2 onward, jerk
        exactly 0.0 from sample 3 onward."""
        n = 6
        timestamps = [float(i) for i in range(n)]
        points = [Point3D(2.0 * i, 0.0, 0.0) for i in range(n)]
        config = DerivativeConfig(max_gap_s=2.0)  # generous: samples are 1s apart here

        result = compute_point_derivatives(timestamps, points, config)

        assert result[0].velocity is None
        for i in range(1, n):
            assert result[i].velocity == pytest.approx(Point3D(2.0, 0.0, 0.0))

        assert result[0].acceleration is None
        assert result[1].acceleration is None
        for i in range(2, n):
            assert result[i].acceleration == pytest.approx(Point3D(0.0, 0.0, 0.0), abs=1e-9)

        assert result[0].jerk is None
        assert result[1].jerk is None
        assert result[2].jerk is None
        for i in range(3, n):
            assert result[i].jerk == pytest.approx(Point3D(0.0, 0.0, 0.0), abs=1e-9)

    def test_constant_acceleration_motion(self) -> None:
        """x(t) = 2t^2 (i.e. constant acceleration a=4), dt=1.
        Backward-difference velocity: v[i] = x[i]-x[i-1] = 4i - 2.
        Backward-difference acceleration of that linear v: exactly 4.0
        from sample 2 onward. Jerk: exactly 0.0 from sample 3 onward."""
        n = 6
        timestamps = [float(i) for i in range(n)]
        points = [Point3D(2.0 * i * i, 0.0, 0.0) for i in range(n)]
        config = DerivativeConfig(max_gap_s=2.0)  # generous: samples are 1s apart here

        result = compute_point_derivatives(timestamps, points, config)

        expected_velocity_x = [None, 2.0, 6.0, 10.0, 14.0, 18.0]
        for i in range(1, n):
            assert result[i].velocity.x == pytest.approx(expected_velocity_x[i])

        for i in range(2, n):
            assert result[i].acceleration.x == pytest.approx(4.0)

        for i in range(3, n):
            assert result[i].jerk.x == pytest.approx(0.0, abs=1e-9)

    def test_none_point_produces_none_sample(self) -> None:
        points = [Point3D(0.0, 0.0, 0.0), None, Point3D(1.0, 0.0, 0.0)]
        timestamps = [0.0, DT, 2 * DT]
        result = compute_point_derivatives(timestamps, points)
        assert result[1] is None

    def test_short_gap_does_not_break_velocity_continuity(self) -> None:
        """A one-frame dropout well under max_gap_s should still let the
        next real detection compute a velocity against the last real one."""
        config = DerivativeConfig(max_gap_s=0.5)
        points = [Point3D(0.0, 0.0, 0.0), None, Point3D(2 * DT, 0.0, 0.0)]
        timestamps = [0.0, DT, 2 * DT]
        result = compute_point_derivatives(timestamps, points, config)
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None
        # velocity = (2*DT - 0) / (2*DT - 0) = 1.0 exactly
        assert result[2].velocity.x == pytest.approx(1.0)

    def test_long_gap_breaks_velocity_continuity(self) -> None:
        """A gap longer than max_gap_s should leave velocity as None right
        after the gap, rather than computing a velocity across it."""
        config = DerivativeConfig(max_gap_s=0.1)
        # ~0.667s gap between index 0 and index 20 at 30fps.
        timestamps = [i * DT for i in range(21)]
        points: list[Point3D | None] = [Point3D(0.0, 0.0, 0.0)]
        points += [None] * 19
        points.append(Point3D(10.0, 0.0, 0.0))

        result = compute_point_derivatives(timestamps, points, config)
        assert result[20] is not None
        assert result[20].velocity is None

    def test_derivatives_recover_after_a_long_gap(self) -> None:
        """After a long gap resets the chain, enough subsequent
        continuous samples should let velocity/acceleration/jerk become
        available again, following the same cascading pattern as at the
        start of a sequence."""
        config = DerivativeConfig(max_gap_s=0.1)
        timestamps = [i * DT for i in range(25)]
        points: list[Point3D | None] = [Point3D(0.0, 0.0, 0.0)]
        points += [None] * 20
        # Four more continuous, evenly-spaced constant-velocity samples
        # after the gap -- enough to re-establish velocity, acceleration,
        # and jerk in turn. 1 + 20 + 4 = 25, matching len(timestamps).
        points += [Point3D(2.0 * i, 0.0, 0.0) for i in range(4)]

        result = compute_point_derivatives(timestamps, points, config)
        # index 21 = first sample after the gap -> velocity None (nothing to diff against)
        assert result[21].velocity is None
        # index 22 = second continuous sample -> velocity available
        assert result[22].velocity is not None
        assert result[22].acceleration is None
        # index 23 -> acceleration available
        assert result[23].acceleration is not None
        assert result[23].jerk is None
        # index 24 -> jerk available
        assert result[24].jerk is not None


class TestComputeKeypointDerivatives:
    def _frame(self, i: int, x: float) -> PoseFrame:
        kp = PoseKeypoint(name=KeypointName.LEFT_WRIST, x=x, y=0.0, z=0.0)
        return PoseFrame(
            frame_index=i, timestamp_s=i * DT, keypoints={KeypointName.LEFT_WRIST: kp}
        )

    def test_extracts_named_keypoint_trajectory(self) -> None:
        frames = [self._frame(i, x=float(i)) for i in range(5)]
        sequence = PoseSequence(
            frames=frames,
            coordinate_space=CoordinateSpace.WORLD_METERS,
            metadata=VideoMetadata(fps=FPS),
        )
        result = compute_keypoint_derivatives(sequence, KeypointName.LEFT_WRIST)
        assert len(result) == 5
        assert result[0].velocity is None
        for i in range(1, 5):
            assert result[i].velocity.x == pytest.approx(1.0 / DT)

    def test_missing_keypoint_produces_none(self) -> None:
        empty_frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={})
        sequence = PoseSequence(
            frames=[empty_frame],
            coordinate_space=CoordinateSpace.WORLD_METERS,
            metadata=VideoMetadata(fps=FPS),
        )
        result = compute_keypoint_derivatives(sequence, KeypointName.LEFT_WRIST)
        assert result[0] is None


class TestComputeComDerivatives:
    def test_extracts_com_trajectory(self) -> None:
        frames = [
            PoseFrame(frame_index=i, timestamp_s=i * DT, keypoints={}) for i in range(4)
        ]
        sequence = PoseSequence(
            frames=frames,
            coordinate_space=CoordinateSpace.WORLD_METERS,
            metadata=VideoMetadata(fps=FPS),
        )
        com_trajectory = [
            CenterOfMassEstimate(x=float(i), y=0.0, z=0.0, mass_coverage=1.0) for i in range(4)
        ]
        result = compute_com_derivatives(sequence, com_trajectory)
        assert len(result) == 4
        assert result[1].velocity.x == pytest.approx(1.0 / DT)

    def test_none_com_estimate_produces_none_sample(self) -> None:
        frames = [
            PoseFrame(frame_index=i, timestamp_s=i * DT, keypoints={}) for i in range(2)
        ]
        sequence = PoseSequence(
            frames=frames,
            coordinate_space=CoordinateSpace.WORLD_METERS,
            metadata=VideoMetadata(fps=FPS),
        )
        com_trajectory: list[CenterOfMassEstimate | None] = [None, None]
        result = compute_com_derivatives(sequence, com_trajectory)
        assert result == [None, None]

    def test_mismatched_lengths_raise(self) -> None:
        frames = [PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={})]
        sequence = PoseSequence(
            frames=frames,
            coordinate_space=CoordinateSpace.WORLD_METERS,
            metadata=VideoMetadata(fps=FPS),
        )
        with pytest.raises(ValueError):
            compute_com_derivatives(sequence, [])