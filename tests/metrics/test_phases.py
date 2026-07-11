"""Tests for climbiq.metrics.phases.

Synthetic KinematicSample sequences with known speed profiles, so expected
phase boundaries can be worked out by hand.
"""

from __future__ import annotations

import pytest

from climbiq.kinematics.com import Point3D
from climbiq.kinematics.derivatives import KinematicSample
from climbiq.metrics.phases import (
    MovementPhase,
    PhaseSegmentationConfig,
    PhaseType,
    segment_phases,
)
from climbiq.pose.types import CoordinateSpace, PoseFrame, PoseSequence, VideoMetadata

FPS = 30.0
DT = 1.0 / FPS


def _sequence(n: int) -> PoseSequence:
    frames = [PoseFrame(frame_index=i, timestamp_s=i * DT, keypoints={}) for i in range(n)]
    return PoseSequence(
        frames=frames,
        coordinate_space=CoordinateSpace.WORLD_METERS,
        metadata=VideoMetadata(fps=FPS),
    )


def _sample(speed: float | None) -> KinematicSample | None:
    """A KinematicSample with a velocity of the given speed along x, or a
    velocity-less sample (still "known", just no speed) if speed is None."""
    position = Point3D(0.0, 0.0, 0.0)
    velocity = Point3D(speed, 0.0, 0.0) if speed is not None else None
    return KinematicSample(position=position, velocity=velocity, acceleration=None, jerk=None)


class TestPhaseSegmentationConfig:
    def test_rejects_non_positive_enter_speed(self) -> None:
        with pytest.raises(ValueError):
            PhaseSegmentationConfig(static_enter_speed=0.0)

    def test_rejects_exit_speed_below_enter_speed(self) -> None:
        with pytest.raises(ValueError):
            PhaseSegmentationConfig(static_enter_speed=0.2, static_exit_speed=0.1)

    def test_rejects_non_positive_min_duration(self) -> None:
        with pytest.raises(ValueError):
            PhaseSegmentationConfig(min_phase_duration_s=0.0)

    def test_accepts_equal_enter_and_exit_speed(self) -> None:
        PhaseSegmentationConfig(static_enter_speed=0.1, static_exit_speed=0.1)  # no raise


class TestSegmentPhases:
    def test_empty_sequence_returns_empty(self) -> None:
        assert segment_phases(_sequence(0), []) == []

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            segment_phases(_sequence(3), [_sample(0.0), _sample(0.0)])

    def test_all_none_samples_produce_one_unknown_segment(self) -> None:
        n = 10
        sequence = _sequence(n)
        kinematics: list[KinematicSample | None] = [None] * n
        result = segment_phases(sequence, kinematics)
        assert len(result) == 1
        assert result[0].phase_type == PhaseType.UNKNOWN
        assert result[0].start_frame_index == 0
        assert result[0].end_frame_index == n - 1

    def test_sustained_low_speed_classified_as_static(self) -> None:
        n = 20
        sequence = _sequence(n)
        kinematics = [_sample(0.01) for _ in range(n)]  # well below default enter/exit
        result = segment_phases(sequence, kinematics)
        assert len(result) == 1
        assert result[0].phase_type == PhaseType.STATIC

    def test_sustained_high_speed_classified_as_moving(self) -> None:
        n = 20
        sequence = _sequence(n)
        kinematics = [_sample(1.0) for _ in range(n)]  # well above default exit
        result = segment_phases(sequence, kinematics)
        assert len(result) == 1
        assert result[0].phase_type == PhaseType.MOVING

    def test_static_then_moving_produces_two_segments(self) -> None:
        static_n, moving_n = 20, 20
        sequence = _sequence(static_n + moving_n)
        kinematics = [_sample(0.01) for _ in range(static_n)] + [
            _sample(1.0) for _ in range(moving_n)
        ]
        result = segment_phases(sequence, kinematics)
        assert [seg.phase_type for seg in result] == [PhaseType.STATIC, PhaseType.MOVING]
        assert result[0].start_frame_index == 0
        assert result[0].end_frame_index == static_n - 1
        assert result[1].start_frame_index == static_n
        assert result[1].end_frame_index == static_n + moving_n - 1

    def test_hysteresis_prevents_flicker_in_the_dead_band(self) -> None:
        """A speed sitting in the dead band between enter and exit
        thresholds should never trigger a transition on its own --
        only crossing the *far* threshold from the current state does."""
        config = PhaseSegmentationConfig(static_enter_speed=0.05, static_exit_speed=0.15)
        n = 30
        sequence = _sequence(n)
        # Start clearly static, then hover in the dead band (0.05-0.15) --
        # should remain STATIC throughout, never flicker to MOVING.
        kinematics = [_sample(0.01) for _ in range(10)] + [_sample(0.10) for _ in range(20)]
        result = segment_phases(sequence, kinematics, config)
        assert len(result) == 1
        assert result[0].phase_type == PhaseType.STATIC

    def test_unknown_gap_is_not_merged_into_neighbors(self) -> None:
        n = 30
        sequence = _sequence(n)
        kinematics = (
            [_sample(1.0) for _ in range(10)]  # MOVING
            + [None for _ in range(10)]  # UNKNOWN gap
            + [_sample(1.0) for _ in range(10)]  # MOVING again
        )
        result = segment_phases(sequence, kinematics)
        assert [seg.phase_type for seg in result] == [
            PhaseType.MOVING,
            PhaseType.UNKNOWN,
            PhaseType.MOVING,
        ]

    def test_short_static_blip_is_merged_into_neighbor(self) -> None:
        """A brief dip below the enter threshold, shorter than
        min_phase_duration_s, should be absorbed rather than becoming its
        own tiny segment."""
        config = PhaseSegmentationConfig(min_phase_duration_s=0.1)  # ~3 frames at 30fps
        n = 40
        sequence = _sequence(n)
        # 15 frames MOVING, 2 frames dip to STATIC-speed (too short to
        # survive min_phase_duration_s), 15 frames MOVING again.
        kinematics = (
            [_sample(1.0) for _ in range(15)]
            + [_sample(0.01) for _ in range(2)]
            + [_sample(1.0) for _ in range(23)]
        )
        result = segment_phases(sequence, kinematics, config)
        assert len(result) == 1
        assert result[0].phase_type == PhaseType.MOVING

    def test_short_segment_between_unknown_neighbors_is_preserved(self) -> None:
        """A short STATIC/MOVING segment with only UNKNOWN neighbors
        should NOT be merged away (that would discard real evidence to
        satisfy a tidiness threshold) -- it's kept even though it's
        shorter than min_phase_duration_s."""
        config = PhaseSegmentationConfig(min_phase_duration_s=1.0)  # deliberately large
        n = 12
        sequence = _sequence(n)
        kinematics = (
            [None for _ in range(5)]  # UNKNOWN
            + [_sample(1.0) for _ in range(2)]  # short MOVING blip
            + [None for _ in range(5)]  # UNKNOWN
        )
        result = segment_phases(sequence, kinematics, config)
        assert [seg.phase_type for seg in result] == [
            PhaseType.UNKNOWN,
            PhaseType.MOVING,
            PhaseType.UNKNOWN,
        ]

    def test_segments_cover_every_frame_exactly_once(self) -> None:
        n = 50
        sequence = _sequence(n)
        kinematics = (
            [_sample(0.01) for _ in range(15)]
            + [None for _ in range(10)]
            + [_sample(1.0) for _ in range(25)]
        )
        result = segment_phases(sequence, kinematics)
        covered_frames: list[int] = []
        for segment in result:
            covered_frames.extend(range(segment.start_frame_index, segment.end_frame_index + 1))
        assert covered_frames == list(range(n))

    def test_duration_s_and_frame_count(self) -> None:
        phase = MovementPhase(
            phase_type=PhaseType.STATIC,
            start_frame_index=10,
            end_frame_index=19,
            start_timestamp_s=10 * DT,
            end_timestamp_s=19 * DT,
        )
        assert phase.frame_count == 10
        assert phase.duration_s == pytest.approx(9 * DT)