"""Tests for climbiq.kinematics.com.

No MediaPipe/OpenCV involved -- synthetic PoseFrame data with known,
hand-computable geometry, so the arithmetic itself can be checked exactly,
not just "some plausible-looking number came out."
"""

from __future__ import annotations

import pytest

from climbiq.kinematics.com import (
    _TOTAL_MODEL_MASS,
    CenterOfMassEstimate,
    compute_center_of_mass_trajectory,
    estimate_center_of_mass,
)
from climbiq.pose.types import (
    CoordinateSpace,
    KeypointName,
    PoseFrame,
    PoseKeypoint,
    PoseSequence,
    VideoMetadata,
)


def _kp(name: KeypointName, x: float, y: float, z: float | None = 0.0) -> PoseKeypoint:
    return PoseKeypoint(name=name, x=x, y=y, z=z)


def _symmetric_standing_frame(z: float | None = 0.0) -> PoseFrame:
    """A perfectly bilaterally-symmetric standing pose, centered on x=0.

    Every left/right pair is mirrored around x=0 at the same y, so the
    overall CoM's x-coordinate must be exactly 0.0 regardless of the
    (correct) mass fractions used -- this makes the test's expected value
    derivable from symmetry alone, not from trusting the module's own
    arithmetic to check itself.
    """
    keypoints = {
        KeypointName.LEFT_SHOULDER: _kp(KeypointName.LEFT_SHOULDER, -1.0, 5.0, z),
        KeypointName.RIGHT_SHOULDER: _kp(KeypointName.RIGHT_SHOULDER, 1.0, 5.0, z),
        KeypointName.LEFT_ELBOW: _kp(KeypointName.LEFT_ELBOW, -1.2, 4.0, z),
        KeypointName.RIGHT_ELBOW: _kp(KeypointName.RIGHT_ELBOW, 1.2, 4.0, z),
        KeypointName.LEFT_WRIST: _kp(KeypointName.LEFT_WRIST, -1.3, 3.0, z),
        KeypointName.RIGHT_WRIST: _kp(KeypointName.RIGHT_WRIST, 1.3, 3.0, z),
        KeypointName.LEFT_INDEX: _kp(KeypointName.LEFT_INDEX, -1.35, 2.8, z),
        KeypointName.RIGHT_INDEX: _kp(KeypointName.RIGHT_INDEX, 1.35, 2.8, z),
        KeypointName.LEFT_HIP: _kp(KeypointName.LEFT_HIP, -0.5, 3.0, z),
        KeypointName.RIGHT_HIP: _kp(KeypointName.RIGHT_HIP, 0.5, 3.0, z),
        KeypointName.LEFT_KNEE: _kp(KeypointName.LEFT_KNEE, -0.5, 1.5, z),
        KeypointName.RIGHT_KNEE: _kp(KeypointName.RIGHT_KNEE, 0.5, 1.5, z),
        KeypointName.LEFT_ANKLE: _kp(KeypointName.LEFT_ANKLE, -0.5, 0.0, z),
        KeypointName.RIGHT_ANKLE: _kp(KeypointName.RIGHT_ANKLE, 0.5, 0.0, z),
        KeypointName.LEFT_HEEL: _kp(KeypointName.LEFT_HEEL, -0.5, -0.1, z),
        KeypointName.RIGHT_HEEL: _kp(KeypointName.RIGHT_HEEL, 0.5, -0.1, z),
        KeypointName.LEFT_FOOT_INDEX: _kp(KeypointName.LEFT_FOOT_INDEX, -0.5, 0.2, z),
        KeypointName.RIGHT_FOOT_INDEX: _kp(KeypointName.RIGHT_FOOT_INDEX, 0.5, 0.2, z),
        KeypointName.NOSE: _kp(KeypointName.NOSE, 0.0, 6.0, z),
    }
    return PoseFrame(frame_index=0, timestamp_s=0.0, keypoints=keypoints)


class TestModelIntegrity:
    def test_total_model_mass_is_approximately_one(self) -> None:
        # Winter's published table sums to ~0.997, not exactly 1.0 --
        # this pins that down as an expected, understood property, not
        # an accidental bug.
        assert _TOTAL_MODEL_MASS == pytest.approx(0.997, abs=0.01)


class TestEstimateCenterOfMass:
    def test_empty_frame_returns_none(self) -> None:
        empty_frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={})
        assert estimate_center_of_mass(empty_frame) is None

    def test_frame_with_no_resolvable_segments_returns_none(self) -> None:
        # Only a single, isolated landmark -- no segment can be built from it.
        kp = _kp(KeypointName.LEFT_EYE, 0.0, 0.0)
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_EYE: kp})
        assert estimate_center_of_mass(frame) is None

    def test_symmetric_pose_has_zero_x_by_symmetry(self) -> None:
        frame = _symmetric_standing_frame()
        estimate = estimate_center_of_mass(frame)
        assert estimate is not None
        assert estimate.x == pytest.approx(0.0, abs=1e-9)

    def test_full_pose_has_full_mass_coverage(self) -> None:
        frame = _symmetric_standing_frame()
        estimate = estimate_center_of_mass(frame)
        assert estimate is not None
        assert estimate.mass_coverage == pytest.approx(1.0)

    def test_com_y_is_between_lowest_and_highest_landmark(self) -> None:
        # Sanity bound: whatever the exact weighting, the CoM shouldn't
        # fall outside the physical extent of the body.
        frame = _symmetric_standing_frame()
        estimate = estimate_center_of_mass(frame)
        assert estimate is not None
        assert -0.1 <= estimate.y <= 6.0

    def test_com_z_present_when_all_contributing_z_present(self) -> None:
        frame = _symmetric_standing_frame(z=2.5)
        estimate = estimate_center_of_mass(frame)
        assert estimate is not None
        assert estimate.z == pytest.approx(2.5, abs=1e-9)

    def test_com_z_none_when_z_missing(self) -> None:
        frame = _symmetric_standing_frame(z=None)
        estimate = estimate_center_of_mass(frame)
        assert estimate is not None
        assert estimate.z is None

    def test_missing_one_limb_reduces_coverage_but_still_estimates(self) -> None:
        full_frame = _symmetric_standing_frame()
        # Drop the right hand/forearm/upper-arm chain entirely.
        reduced_keypoints = {
            name: kp
            for name, kp in full_frame.keypoints.items()
            if name
            not in (
                KeypointName.RIGHT_SHOULDER,
                KeypointName.RIGHT_ELBOW,
                KeypointName.RIGHT_WRIST,
                KeypointName.RIGHT_INDEX,
            )
        }
        # left_upper_arm and left_forearm and left_hand still resolvable;
        # trunk and head_neck need RIGHT_SHOULDER too, so those drop out.
        reduced_frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints=reduced_keypoints)

        full_estimate = estimate_center_of_mass(full_frame)
        reduced_estimate = estimate_center_of_mass(reduced_frame)

        assert full_estimate is not None
        assert reduced_estimate is not None
        assert reduced_estimate.mass_coverage < full_estimate.mass_coverage
        assert reduced_estimate.mass_coverage > 0.0

    def test_removing_all_landmarks_for_a_bilateral_pair_shifts_com_toward_remaining_side(
        self,
    ) -> None:
        full_frame = _symmetric_standing_frame()
        keypoints_without_right_arm_chain = {
            name: kp
            for name, kp in full_frame.keypoints.items()
            if name
            not in (
                KeypointName.RIGHT_SHOULDER,
                KeypointName.RIGHT_ELBOW,
                KeypointName.RIGHT_WRIST,
                KeypointName.RIGHT_INDEX,
            )
        }
        frame = PoseFrame(
            frame_index=0, timestamp_s=0.0, keypoints=keypoints_without_right_arm_chain
        )
        estimate = estimate_center_of_mass(frame)
        assert estimate is not None
        # Left arm keypoints are all at negative x; with the mirrored right
        # arm gone, the estimate should shift negative (left) from the
        # perfectly-symmetric x=0.
        assert estimate.x < 0.0


class TestComputeCenterOfMassTrajectory:
    def test_result_is_index_aligned_with_sequence(self) -> None:
        frames = [
            PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={}),
            _symmetric_standing_frame(),
        ]
        sequence = PoseSequence(
            frames=[
                PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={}),
                PoseFrame(
                    frame_index=1, timestamp_s=1 / 30, keypoints=frames[1].keypoints
                ),
            ],
            coordinate_space=CoordinateSpace.WORLD_METERS,
            metadata=VideoMetadata(fps=30.0),
        )
        trajectory = compute_center_of_mass_trajectory(sequence)
        assert len(trajectory) == len(sequence)
        assert trajectory[0] is None
        assert isinstance(trajectory[1], CenterOfMassEstimate)