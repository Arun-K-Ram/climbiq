"""Temporal smoothing for pose sequences.

Raw per-frame pose estimates are jittery even when the underlying motion is
smooth -- every downstream kinematic computation (velocity, jerk, center of
mass) amplifies that jitter through differentiation. This module removes it
before anything else touches the data.

Responsibility of this module, and only this module: take a PoseSequence,
return a PoseSequence with smoothed coordinates. It does not fabricate
keypoints for frames where nothing was detected -- see `smooth_pose_sequence`
docstring for why, and how gaps are handled instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from climbiq.pose.types import KeypointName, PoseFrame, PoseKeypoint, PoseSequence

_AXES: tuple[str, ...] = ("x", "y", "z")


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    """Tuning parameters for the One-Euro filter and gap handling.

    The One-Euro filter (Casiez et al., 2012) is a low-pass filter whose
    cutoff frequency adapts to the signal's speed: it smooths aggressively
    when a keypoint is nearly still (removing jitter) and relaxes when it's
    moving fast (avoiding the lag a fixed cutoff would introduce on a
    dynamic move -- a real concern for climbing, where a lagging filter
    would blur exactly the explosive movements most worth analyzing).

    Attributes:
        min_cutoff: Base cutoff frequency (Hz) used when a keypoint is
            stationary. Lower values mean more smoothing of jitter, at the
            cost of a small amount of lag on slow movements. This is the
            main "how much smoothing" knob.
        beta: How much the cutoff frequency increases with speed. Higher
            values reduce lag on fast movements but let more jitter
            through while moving. 0.0 disables speed-adaptiveness
            entirely (a plain fixed-cutoff low-pass filter).
        d_cutoff: Cutoff frequency used when smoothing the *velocity*
            estimate that drives the adaptive cutoff above. Rarely needs
            tuning; the filter's original authors treat 1.0 as a
            reasonable default across use cases.
        max_gap_s: Time gap between consecutive detections of the same
            keypoint beyond which the filter resets instead of smoothing
            through it. Below this threshold, a short dropout (a missed
            detection for a frame or two -- common from motion blur or
            momentary occlusion) does not disrupt the filter's internal
            state, so the next detection blends smoothly with pre-gap
            history. Above it (e.g. the climber leaves frame, or a long
            occlusion), the next detection is emitted unsmoothed rather
            than being blended with data from before a gap that long --
            treating it as noise-continuity would fabricate a movement
            trend that was never observed.

    These defaults are reasonable starting points, not validated against
    real climbing footage yet -- expect to tune `min_cutoff` and `beta`
    empirically once there's a labeled or eyeballed reference to compare
    smoothed trajectories against (this is exactly the kind of decision
    the Phase 0 data-collection pass in the project roadmap exists for).
    """

    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0
    max_gap_s: float = 0.5

    def __post_init__(self) -> None:
        for field_name, value in (
            ("min_cutoff", self.min_cutoff),
            ("d_cutoff", self.d_cutoff),
            ("max_gap_s", self.max_gap_s),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be > 0, got {value!r}")
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta!r}")


class _OneEuroFilter:
    """Single-scalar One-Euro filter.

    Stateful and sequential by design: call `.filter()` once per observed
    sample, in increasing timestamp order. Not thread-safe, not meant to
    be -- one instance is created per (keypoint, axis) pair and used only
    from the single sequential pass in `smooth_pose_sequence`.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._initialized = False
        self._prev_value = 0.0
        self._prev_derivative = 0.0
        self._prev_timestamp_s = 0.0

    def reset(self) -> None:
        """Discard filter history. The next `.filter()` call passes through raw."""
        self._initialized = False

    def filter(self, value: float, timestamp_s: float) -> float:
        if not self._initialized:
            self._prev_value = value
            self._prev_derivative = 0.0
            self._prev_timestamp_s = timestamp_s
            self._initialized = True
            return value

        dt = timestamp_s - self._prev_timestamp_s
        if dt <= 0:
            # Non-increasing timestamps shouldn't occur given PoseSequence's
            # ordering invariant, but guard rather than divide by zero if
            # this filter is ever reused outside that guarantee.
            return self._prev_value

        raw_derivative = (value - self._prev_value) / dt
        derivative_alpha = self._smoothing_factor(dt, self._d_cutoff)
        smoothed_derivative = self._exponential_smoothing(
            raw_derivative, self._prev_derivative, derivative_alpha
        )

        adaptive_cutoff = self._min_cutoff + self._beta * abs(smoothed_derivative)
        value_alpha = self._smoothing_factor(dt, adaptive_cutoff)
        smoothed_value = self._exponential_smoothing(value, self._prev_value, value_alpha)

        self._prev_value = smoothed_value
        self._prev_derivative = smoothed_derivative
        self._prev_timestamp_s = timestamp_s
        return smoothed_value

    @staticmethod
    def _smoothing_factor(dt: float, cutoff_hz: float) -> float:
        time_constant = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + time_constant / dt)

    @staticmethod
    def _exponential_smoothing(value: float, previous: float, alpha: float) -> float:
        return alpha * value + (1.0 - alpha) * previous


def smooth_pose_sequence(
    sequence: PoseSequence, config: SmoothingConfig | None = None
) -> PoseSequence:
    """Apply One-Euro temporal smoothing to every keypoint in a PoseSequence.

    Frames where no person was detected (`frame.is_empty`) pass through
    unchanged -- smoothing never fabricates a keypoint for a frame with no
    underlying detection. See `SmoothingConfig.max_gap_s` for how gaps
    between detections are handled instead.

    Each keypoint (e.g. LEFT_WRIST) is smoothed independently per axis,
    using its own filter state. This is deliberately per-keypoint, not
    per-frame: if a future pose backend ever reports some keypoints but not
    others in a given frame (this one doesn't -- MediaPipe returns all 33
    landmarks or none), each keypoint's gap-handling still behaves
    correctly in isolation rather than being dictated by whichever
    keypoint happened to drop out.

    Args:
        sequence: The PoseSequence to smooth. Not mutated.
        config: Filter tuning and gap-handling parameters. Uses
            SmoothingConfig()'s defaults if not provided.

    Returns:
        A new PoseSequence with the same frame count, ordering, metadata,
        and coordinate_space as the input, and smoothed keypoint
        coordinates. visibility/presence scores are carried through
        unchanged from the input -- smoothing affects position, not the
        model's confidence about that position.
    """
    if config is None:
        config = SmoothingConfig()

    if len(sequence) == 0:
        return sequence

    filters: dict[tuple[KeypointName, str], _OneEuroFilter] = {}
    last_seen_timestamp_s: dict[KeypointName, float] = {}

    smoothed_frames: list[PoseFrame] = []

    for frame in sequence:
        if frame.is_empty:
            smoothed_frames.append(frame)
            continue

        smoothed_keypoints: dict[KeypointName, PoseKeypoint] = {}
        for name, keypoint in frame.keypoints.items():
            last_t = last_seen_timestamp_s.get(name)
            if last_t is not None and (frame.timestamp_s - last_t) > config.max_gap_s:
                for axis in _AXES:
                    filters.pop((name, axis), None)

            smoothed_x = _filter_for(filters, name, "x", config).filter(
                keypoint.x, frame.timestamp_s
            )
            smoothed_y = _filter_for(filters, name, "y", config).filter(
                keypoint.y, frame.timestamp_s
            )
            smoothed_z = (
                _filter_for(filters, name, "z", config).filter(keypoint.z, frame.timestamp_s)
                if keypoint.z is not None
                else None
            )

            smoothed_keypoints[name] = PoseKeypoint(
                name=name,
                x=smoothed_x,
                y=smoothed_y,
                z=smoothed_z,
                visibility=keypoint.visibility,
                presence=keypoint.presence,
            )
            last_seen_timestamp_s[name] = frame.timestamp_s

        smoothed_frames.append(
            PoseFrame(
                frame_index=frame.frame_index,
                timestamp_s=frame.timestamp_s,
                keypoints=smoothed_keypoints,
                detection_confidence=frame.detection_confidence,
            )
        )

    return PoseSequence(
        frames=smoothed_frames,
        coordinate_space=sequence.coordinate_space,
        metadata=sequence.metadata,
        estimator_name=sequence.estimator_name,
    )


def _filter_for(
    filters: dict[tuple[KeypointName, str], _OneEuroFilter],
    name: KeypointName,
    axis: str,
    config: SmoothingConfig,
) -> _OneEuroFilter:
    key = (name, axis)
    existing = filters.get(key)
    if existing is not None:
        return existing
    created = _OneEuroFilter(config.min_cutoff, config.beta, config.d_cutoff)
    filters[key] = created
    return created