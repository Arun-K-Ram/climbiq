"""Tests for climbiq.metrics.pauses."""

from __future__ import annotations

import pytest

from climbiq.metrics.pauses import PauseDetectionConfig, detect_pauses, summarize_pauses
from climbiq.metrics.phases import MovementPhase, PhaseType

FPS = 30.0
DT = 1.0 / FPS


def _phase(phase_type: PhaseType, start: int, end: int) -> MovementPhase:
    return MovementPhase(
        phase_type=phase_type,
        start_frame_index=start,
        end_frame_index=end,
        start_timestamp_s=start * DT,
        end_timestamp_s=end * DT,
    )


class TestPauseDetectionConfig:
    def test_rejects_non_positive_min_duration(self) -> None:
        with pytest.raises(ValueError):
            PauseDetectionConfig(min_pause_duration_s=0.0)


class TestDetectPauses:
    def test_empty_phases_returns_empty(self) -> None:
        assert detect_pauses([]) == []

    def test_short_static_phase_is_not_a_pause(self) -> None:
        # 5 frames at 30fps ~= 0.13s, well under the 1.0s default threshold
        phases = [_phase(PhaseType.STATIC, 0, 4)]
        assert detect_pauses(phases) == []

    def test_long_static_phase_is_a_pause(self) -> None:
        # 60 frames at 30fps = ~2s, over the default 1.0s threshold
        phases = [_phase(PhaseType.STATIC, 0, 59)]
        result = detect_pauses(phases)
        assert len(result) == 1
        assert result[0].start_frame_index == 0
        assert result[0].end_frame_index == 59

    def test_moving_phase_is_never_a_pause_regardless_of_duration(self) -> None:
        phases = [_phase(PhaseType.MOVING, 0, 200)]  # long, but MOVING
        assert detect_pauses(phases) == []

    def test_unknown_phase_is_never_a_pause_regardless_of_duration(self) -> None:
        phases = [_phase(PhaseType.UNKNOWN, 0, 200)]  # long, but UNKNOWN
        assert detect_pauses(phases) == []

    def test_custom_threshold(self) -> None:
        config = PauseDetectionConfig(min_pause_duration_s=0.05)
        phases = [_phase(PhaseType.STATIC, 0, 4)]  # ~0.13s
        result = detect_pauses(phases, config)
        assert len(result) == 1

    def test_multiple_pauses_in_chronological_order(self) -> None:
        phases = [
            _phase(PhaseType.STATIC, 0, 59),
            _phase(PhaseType.MOVING, 60, 70),
            _phase(PhaseType.STATIC, 71, 150),
        ]
        result = detect_pauses(phases)
        assert len(result) == 2
        assert result[0].start_frame_index == 0
        assert result[1].start_frame_index == 71


class TestSummarizePauses:
    def test_empty_list_has_no_longest_pause(self) -> None:
        summary = summarize_pauses([])
        assert summary.pauses == []
        assert summary.total_pause_time_s == 0.0
        assert summary.longest_pause is None

    def test_totals_and_longest(self) -> None:
        phases = [
            _phase(PhaseType.STATIC, 0, 29),  # ~0.97s
            _phase(PhaseType.STATIC, 100, 199),  # ~3.3s -- longest
        ]
        pauses = detect_pauses(phases, PauseDetectionConfig(min_pause_duration_s=0.5))
        summary = summarize_pauses(pauses)
        assert len(summary.pauses) == 2
        assert summary.total_pause_time_s == pytest.approx(
            pauses[0].duration_s + pauses[1].duration_s
        )
        assert summary.longest_pause == pauses[1]