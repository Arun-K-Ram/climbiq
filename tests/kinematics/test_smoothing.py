"""Tests for climbiq.kinematics.smoothing.

No MediaPipe/OpenCV involved -- all synthetic PoseSequence data, so this
runs instantly in any environment.
"""

from __future__ import annotations

import pytest

from climbiq.kinematics.smoothing import SmoothingConfig, smooth_pose_sequence
from climbiq.pose.types import (
    CoordinateSpace,
    KeypointName,
    PoseFrame,
    PoseKeypoint,
    PoseSequence,
    VideoMetadata,
)

FPS = 30.0


def _kp(x: float, y: float, z: float = 0.0) -> PoseKeypoint:
    return PoseKeypoint(name=KeypointName.LEFT_WRIST, x=x, y=y, z=z, visibility=0.9)


def _frame(frame_index: int, x: float | None, y: float | None = None) -> PoseFrame:
    """Build a frame with a single LEFT_WRIST keypoint, or empty if x is None."""
    timestamp_s = frame_index / FPS
    if x is None:
        return PoseFrame(frame_index=frame_index, timestamp_s=timestamp_s, keypoints={})
    return PoseFrame(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        keypoints={KeypointName.LEFT_WRIST: _kp(x, y if y is not None else 0.0)},
    )


def _sequence(frames: list[PoseFrame]) -> PoseSequence:
    return PoseSequence(
        frames=frames,
        coordinate_space=CoordinateSpace.WORLD_METERS,
        metadata=VideoMetadata(fps=FPS),
        estimator_name="synthetic",
    )


class TestSmoothingConfig:
    @pytest.mark.parametrize("field_name", ["min_cutoff", "d_cutoff", "max_gap_s"])
    def test_rejects_non_positive_values(self, field_name: str) -> None:
        with pytest.raises(ValueError):
            SmoothingConfig(**{field_name: 0.0})

    def test_rejects_negative_beta(self) -> None:
        with pytest.raises(ValueError):
            SmoothingConfig(beta=-0.1)

    def test_accepts_defaults(self) -> None:
        SmoothingConfig()  # should not raise


class TestSmoothPoseSequence:
    def test_empty_sequence_returns_unchanged(self) -> None:
        seq = _sequence([])
        assert smooth_pose_sequence(seq) is seq

    def test_preserves_frame_count_and_ordering(self) -> None:
        frames = [_frame(i, x=float(i)) for i in range(10)]
        seq = _sequence(frames)
        smoothed = smooth_pose_sequence(seq)
        assert len(smoothed) == len(seq)
        assert [f.frame_index for f in smoothed] == [f.frame_index for f in seq]
        assert [f.timestamp_s for f in smoothed] == [f.timestamp_s for f in seq]

    def test_preserves_metadata_and_coordinate_space(self) -> None:
        seq = _sequence([_frame(0, x=0.0)])
        smoothed = smooth_pose_sequence(seq)
        assert smoothed.metadata == seq.metadata
        assert smoothed.coordinate_space == seq.coordinate_space
        assert smoothed.estimator_name == seq.estimator_name

    def test_empty_frames_pass_through_unchanged(self) -> None:
        frames = [_frame(0, x=0.0), _frame(1, x=None), _frame(2, x=1.0)]
        seq = _sequence(frames)
        smoothed = smooth_pose_sequence(seq)
        assert smoothed[1].is_empty
        assert smoothed[1] is frames[1]

    def test_constant_signal_is_unchanged_by_filter(self) -> None:
        """A perfectly constant, noise-free signal should pass through
        the filter essentially exactly -- there's nothing to smooth."""
        frames = [_frame(i, x=5.0, y=-2.0) for i in range(20)]
        seq = _sequence(frames)
        smoothed = smooth_pose_sequence(seq)
        for frame in smoothed:
            kp = frame.get(KeypointName.LEFT_WRIST)
            assert kp is not None
            assert kp.x == pytest.approx(5.0, abs=1e-9)
            assert kp.y == pytest.approx(-2.0, abs=1e-9)

    def test_first_sample_passes_through_raw(self) -> None:
        frames = [_frame(0, x=3.0)]
        seq = _sequence(frames)
        smoothed = smooth_pose_sequence(seq)
        kp = smoothed[0].get(KeypointName.LEFT_WRIST)
        assert kp is not None
        assert kp.x == pytest.approx(3.0)

    def test_reduces_jitter_on_noisy_constant_signal(self) -> None:
        """A signal that's constant except for alternating +/- noise should
        have its smoothed variance well below the raw variance."""
        base = 1.0
        noise = 0.05
        raw_values = [base + (noise if i % 2 == 0 else -noise) for i in range(40)]
        frames = [_frame(i, x=raw_values[i]) for i in range(40)]
        seq = _sequence(frames)
        smoothed = smooth_pose_sequence(seq, SmoothingConfig(min_cutoff=0.5, beta=0.0))

        smoothed_values = [
            f.get(KeypointName.LEFT_WRIST).x  # type: ignore[union-attr]
            for f in smoothed
        ]

        def variance(values: list[float]) -> float:
            mean = sum(values) / len(values)
            return sum((v - mean) ** 2 for v in values) / len(values)

        # skip the first few samples while the filter is still converging
        assert variance(smoothed_values[10:]) < variance(raw_values[10:]) * 0.5

    def test_short_gap_does_not_reset_filter_state(self) -> None:
        """A gap shorter than max_gap_s should let the filter keep
        blending with pre-gap history, rather than passing the
        post-gap value through raw."""
        config = SmoothingConfig(min_cutoff=0.1, beta=0.0, max_gap_s=1.0)
        # Several frames to let the filter converge near 0.0, then a
        # 1-frame gap (well under max_gap_s), then a jump to 10.0.
        frames = [_frame(i, x=0.0) for i in range(5)]
        frames.append(_frame(5, x=None))  # short gap: one missed frame
        frames.append(_frame(6, x=10.0))
        seq = _sequence(frames)

        smoothed = smooth_pose_sequence(seq, config)
        kp_after_gap = smoothed[6].get(KeypointName.LEFT_WRIST)
        assert kp_after_gap is not None
        # Filter state carried through -> heavily damped, nowhere near 10.0
        assert kp_after_gap.x < 5.0

    def test_long_gap_resets_filter_state(self) -> None:
        """A gap longer than max_gap_s should reset the filter, so the
        first detection after the gap passes through raw/unsmoothed."""
        config = SmoothingConfig(min_cutoff=0.1, beta=0.0, max_gap_s=0.1)
        frames = [_frame(i, x=0.0) for i in range(5)]  # converges near 0.0
        # Gap of many empty frames -- well beyond max_gap_s at 30fps.
        frames.extend(_frame(i, x=None) for i in range(5, 20))
        frames.append(_frame(20, x=10.0))
        seq = _sequence(frames)

        smoothed = smooth_pose_sequence(seq, config)
        kp_after_gap = smoothed[20].get(KeypointName.LEFT_WRIST)
        assert kp_after_gap is not None
        # Filter reset -> raw passthrough, exactly 10.0
        assert kp_after_gap.x == pytest.approx(10.0)

    def test_visibility_and_presence_pass_through_unchanged(self) -> None:
        kp = PoseKeypoint(
            name=KeypointName.LEFT_WRIST, x=1.0, y=1.0, visibility=0.42, presence=0.77
        )
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp})
        seq = _sequence([frame])
        smoothed = smooth_pose_sequence(seq)
        smoothed_kp = smoothed[0].get(KeypointName.LEFT_WRIST)
        assert smoothed_kp is not None
        assert smoothed_kp.visibility == 0.42
        assert smoothed_kp.presence == 0.77

    def test_none_z_stays_none(self) -> None:
        kp = PoseKeypoint(name=KeypointName.LEFT_WRIST, x=1.0, y=1.0, z=None)
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp})
        seq = _sequence([frame])
        smoothed = smooth_pose_sequence(seq)
        assert smoothed[0].get(KeypointName.LEFT_WRIST).z is None  # type: ignore[union-attr]

    def test_does_not_mutate_input_sequence(self) -> None:
        frames = [_frame(i, x=float(i)) for i in range(5)]
        seq = _sequence(frames)
        original_x_values = [
            f.get(KeypointName.LEFT_WRIST).x for f in seq  # type: ignore[union-attr]
        ]
        smooth_pose_sequence(seq, SmoothingConfig(min_cutoff=0.1))
        after_x_values = [
            f.get(KeypointName.LEFT_WRIST).x for f in seq  # type: ignore[union-attr]
        ]
        assert original_x_values == after_x_values

    def test_multiple_keypoints_are_filtered_independently(self) -> None:
        wrist = PoseKeypoint(name=KeypointName.LEFT_WRIST, x=0.0, y=0.0)
        ankle = PoseKeypoint(name=KeypointName.LEFT_ANKLE, x=100.0, y=100.0)
        frames = [
            PoseFrame(
                frame_index=i,
                timestamp_s=i / FPS,
                keypoints={KeypointName.LEFT_WRIST: wrist, KeypointName.LEFT_ANKLE: ankle},
            )
            for i in range(5)
        ]
        seq = _sequence(frames)
        smoothed = smooth_pose_sequence(seq)
        last = smoothed[-1]
        assert last.get(KeypointName.LEFT_WRIST).x == pytest.approx(0.0)  # type: ignore[union-attr]
        assert last.get(KeypointName.LEFT_ANKLE).x == pytest.approx(100.0)  # type: ignore[union-attr]