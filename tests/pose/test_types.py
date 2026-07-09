"""Tests for climbiq.pose.types.

These tests exercise only pose/types.py and pose/base.py, both of which
have zero external dependencies -- they should run instantly, with no
MediaPipe/OpenCV involved, in any environment.
"""

from __future__ import annotations

import pytest

from climbiq.pose.base import PoseEstimator
from climbiq.pose.types import (
    CoordinateSpace,
    KeypointName,
    PoseFrame,
    PoseKeypoint,
    PoseSequence,
    VideoMetadata,
)


class TestPoseKeypoint:
    def test_confidence_prefers_visibility_over_presence(self) -> None:
        kp = PoseKeypoint(
            name=KeypointName.LEFT_WRIST, x=0.1, y=0.2, visibility=0.9, presence=0.95
        )
        assert kp.confidence == 0.9

    def test_confidence_falls_back_to_presence(self) -> None:
        kp = PoseKeypoint(name=KeypointName.LEFT_WRIST, x=0.1, y=0.2, presence=0.7)
        assert kp.confidence == 0.7

    def test_confidence_is_none_when_both_missing(self) -> None:
        kp = PoseKeypoint(name=KeypointName.LEFT_WRIST, x=0.1, y=0.2)
        assert kp.confidence is None

    @pytest.mark.parametrize("field_name", ["visibility", "presence"])
    def test_rejects_out_of_range_scores(self, field_name: str) -> None:
        with pytest.raises(ValueError):
            PoseKeypoint(name=KeypointName.NOSE, x=0.0, y=0.0, **{field_name: 1.5})


class TestPoseFrame:
    def _keypoint(self) -> PoseKeypoint:
        return PoseKeypoint(name=KeypointName.LEFT_WRIST, x=0.1, y=0.2, visibility=0.9)

    def test_get_returns_keypoint_when_present(self) -> None:
        kp = self._keypoint()
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp})
        assert frame.get(KeypointName.LEFT_WRIST) is kp

    def test_get_returns_none_when_missing(self) -> None:
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={})
        assert frame.get(KeypointName.LEFT_WRIST) is None

    def test_get_applies_min_confidence_gate(self) -> None:
        kp = self._keypoint()  # visibility=0.9
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp})
        assert frame.get(KeypointName.LEFT_WRIST, min_confidence=0.5) is kp
        assert frame.get(KeypointName.LEFT_WRIST, min_confidence=0.95) is None

    def test_contains(self) -> None:
        kp = self._keypoint()
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp})
        assert KeypointName.LEFT_WRIST in frame
        assert KeypointName.RIGHT_WRIST not in frame

    def test_is_empty(self) -> None:
        assert PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={}).is_empty
        kp = self._keypoint()
        assert not PoseFrame(
            frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp}
        ).is_empty

    def test_keypoints_mapping_is_read_only(self) -> None:
        kp = self._keypoint()
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp})
        with pytest.raises(TypeError):
            frame.keypoints[KeypointName.NOSE] = kp  # type: ignore[index]

    def test_frame_is_frozen(self) -> None:
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={})
        with pytest.raises(AttributeError):
            frame.frame_index = 5  # type: ignore[misc]

    def test_rejects_negative_frame_index(self) -> None:
        with pytest.raises(ValueError):
            PoseFrame(frame_index=-1, timestamp_s=0.0, keypoints={})

    def test_rejects_negative_timestamp(self) -> None:
        with pytest.raises(ValueError):
            PoseFrame(frame_index=0, timestamp_s=-0.1, keypoints={})


class TestVideoMetadata:
    def test_rejects_non_positive_fps(self) -> None:
        with pytest.raises(ValueError):
            VideoMetadata(fps=0.0)

    def test_accepts_minimal_metadata(self) -> None:
        meta = VideoMetadata(fps=30.0)
        assert meta.width is None
        assert meta.source_path is None


class TestPoseSequence:
    def _sequence(self, n: int = 5) -> PoseSequence:
        kp = PoseKeypoint(name=KeypointName.LEFT_WRIST, x=0.1, y=0.2, visibility=0.9)
        frames = [
            PoseFrame(
                frame_index=i,
                timestamp_s=i / 30.0,
                # frame 1 has nobody detected, to exercise detected_fraction
                keypoints={} if i == 1 else {KeypointName.LEFT_WRIST: kp},
            )
            for i in range(n)
        ]
        metadata = VideoMetadata(fps=30.0, width=1920, height=1080, source_path="climb.mp4")
        return PoseSequence(
            frames=frames,
            coordinate_space=CoordinateSpace.WORLD_METERS,
            metadata=metadata,
            estimator_name="test_estimator",
        )

    def test_len_and_indexing(self) -> None:
        seq = self._sequence()
        assert len(seq) == 5
        assert seq[2].frame_index == 2

    def test_iteration(self) -> None:
        seq = self._sequence()
        assert [f.frame_index for f in seq] == [0, 1, 2, 3, 4]

    def test_duration_s(self) -> None:
        seq = self._sequence()
        assert seq.duration_s == pytest.approx(4 / 30)

    def test_fps_passthrough(self) -> None:
        assert self._sequence().fps == 30.0

    def test_frame_at_time_nearest_neighbor(self) -> None:
        seq = self._sequence()
        target = 2 / 30.0 + 0.001
        assert seq.frame_at_time(target).frame_index == 2

    def test_frame_at_time_clamps_to_bounds(self) -> None:
        seq = self._sequence()
        assert seq.frame_at_time(-10.0).frame_index == 0
        assert seq.frame_at_time(10_000.0).frame_index == 4

    def test_frame_at_time_on_empty_sequence_raises(self) -> None:
        metadata = VideoMetadata(fps=30.0)
        empty_seq = PoseSequence(
            frames=[], coordinate_space=CoordinateSpace.WORLD_METERS, metadata=metadata
        )
        with pytest.raises(IndexError):
            empty_seq.frame_at_time(0.0)

    def test_detected_fraction(self) -> None:
        seq = self._sequence()
        assert seq.detected_fraction() == pytest.approx(4 / 5)

    def test_rejects_non_increasing_frame_index(self) -> None:
        kp = PoseKeypoint(name=KeypointName.LEFT_WRIST, x=0.1, y=0.2)
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp})
        metadata = VideoMetadata(fps=30.0)
        with pytest.raises(ValueError):
            PoseSequence(
                frames=[frame, frame],
                coordinate_space=CoordinateSpace.WORLD_METERS,
                metadata=metadata,
            )

    def test_coerces_frames_to_tuple(self) -> None:
        """Passing a list should not leave the sequence mutable via that list."""
        kp = PoseKeypoint(name=KeypointName.LEFT_WRIST, x=0.1, y=0.2)
        frame = PoseFrame(frame_index=0, timestamp_s=0.0, keypoints={KeypointName.LEFT_WRIST: kp})
        original_list = [frame]
        seq = PoseSequence(
            frames=original_list,
            coordinate_space=CoordinateSpace.WORLD_METERS,
            metadata=VideoMetadata(fps=30.0),
        )
        original_list.append(frame)
        assert len(seq) == 1  # mutating the original list after construction has no effect


class TestPoseEstimatorInterface:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            PoseEstimator()  # type: ignore[abstract]

    def test_subclass_must_implement_full_contract(self) -> None:
        class IncompleteEstimator(PoseEstimator):
            @property
            def name(self) -> str:
                return "incomplete"
            # missing estimate() -- should still be abstract

        with pytest.raises(TypeError):
            IncompleteEstimator()  # type: ignore[abstract]
