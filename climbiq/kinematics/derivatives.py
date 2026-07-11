"""Velocity, acceleration, and jerk via gap-aware finite differences.

Generic over *any* time-indexed sequence of points -- a single keypoint's
trajectory, the whole-body center-of-mass trajectory from
`climbiq.kinematics.com`, or anything else with the same shape. The core
function (`compute_point_derivatives`) knows nothing about pose data at
all; `compute_keypoint_derivatives` and `compute_com_derivatives` are thin
adapters on top of it for the two trajectory sources ClimbIQ currently has.

Method: backward (causal) finite differences, computed by differentiating
the same way at each order -- velocity is the finite difference of
position, acceleration is the finite difference of velocity, jerk is the
finite difference of acceleration. Backward differences are simpler and
strictly causal (each value depends only on the past, never a future
sample) compared to central differences, at the cost of being slightly
noisier and one-sample-lagged; given that the position data has already
been through `kinematics.smoothing` by the time it reaches here, that
trade-off favors simplicity. Revisit only if real analysis shows the lag
matters (e.g. for precise crux-timing), not preemptively.

Responsibility of this module, and only this module: point trajectories in,
derivative trajectories out. No smoothing (that already happened upstream),
no phase segmentation, no thresholding "is this a pause" -- those are
metrics-layer decisions that consume this module's output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from climbiq.kinematics.com import CenterOfMassEstimate, Point3D
from climbiq.pose.types import KeypointName, PoseKeypoint, PoseSequence


@dataclass(frozen=True, slots=True)
class DerivativeConfig:
    """Gap-handling parameter for derivative computation.

    Attributes:
        max_gap_s: Time gap between two consecutive available points
            beyond which a derivative is *not* computed across that gap.
            Mirrors `SmoothingConfig.max_gap_s` in spirit -- a derivative
            computed across a multi-second dropout would represent a
            movement trend that was never observed, the same failure mode
            gap-aware smoothing exists to avoid. Below this threshold, a
            short dropout (motion blur, momentary occlusion) doesn't
            break the derivative chain; above it, samples immediately
            after the gap get `None` until enough fresh, continuous
            history accumulates (progressively more for velocity, then
            acceleration, then jerk -- see `KinematicSample`).
    """

    max_gap_s: float = 0.5

    def __post_init__(self) -> None:
        if self.max_gap_s <= 0:
            raise ValueError(f"max_gap_s must be > 0, got {self.max_gap_s!r}")


@dataclass(frozen=True, slots=True)
class KinematicSample:
    """Position and its derivatives at one point in time.

    Attributes:
        position: The observed point at this sample. Always present --
            this class is only ever constructed for samples that had a
            real position; see the module-level `compute_point_derivatives`
            docstring for how missing samples are represented instead
            (as `None` in the result list, not as a `KinematicSample`
            with a missing position).
        velocity: Rate of change of position, or None if there wasn't a
            valid prior sample within `DerivativeConfig.max_gap_s` to
            difference against (the very first sample overall, or the
            first sample after a large gap).
        acceleration: Rate of change of velocity, or None under the same
            condition applied one order up -- since it needs two
            consecutive valid velocities, it stays None for one extra
            sample beyond wherever velocity itself starts being available.
        jerk: Rate of change of acceleration, None for one further sample
            beyond that, for the same reason.

    This cascading None pattern is intentional, not a bug to work around:
    it's an honest reflection of how much continuous history is actually
    behind each value. A metrics module computing a "smoothness" score
    from jerk should skip None entries, not treat them as zero.
    """

    position: Point3D
    velocity: Point3D | None
    acceleration: Point3D | None
    jerk: Point3D | None


def magnitude(point: Point3D) -> float:
    """Euclidean norm of a Point3D, e.g. to turn a velocity vector into a scalar speed.

    Treats a missing z (None) as simply absent from the sum, not as zero
    contributing noise -- which happens to be arithmetically identical to
    treating it as 0.0, so 2D and 3D points both get their correct
    magnitude with no special-casing needed.
    """
    z = point.z if point.z is not None else 0.0
    return math.sqrt(point.x**2 + point.y**2 + z**2)


def compute_point_derivatives(
    timestamps: Sequence[float],
    points: Sequence[Point3D | None],
    config: DerivativeConfig | None = None,
) -> list[KinematicSample | None]:
    """Compute velocity/acceleration/jerk for a generic point trajectory.

    Args:
        timestamps: Timestamp in seconds for each sample, strictly
            increasing (this is not validated here -- both current callers,
            PoseSequence-derived timestamps, already guarantee it).
        points: The observed point at each timestamp, or None where no
            observation exists (e.g. the pose estimator detected no
            person in that frame). Must be the same length as `timestamps`.
        config: Gap-handling parameters. Uses DerivativeConfig()'s
            defaults if not provided.

    Returns:
        A list the same length as `points`, index-aligned with it.
        result[i] is None wherever points[i] is None (nothing was
        observed, so there's nothing to report a derivative *of*);
        otherwise it's a KinematicSample whose velocity/acceleration/jerk
        may themselves individually be None per the gap/history rules
        described on KinematicSample.

    Raises:
        ValueError: if `timestamps` and `points` differ in length.
    """
    if config is None:
        config = DerivativeConfig()
    if len(timestamps) != len(points):
        raise ValueError(
            f"timestamps and points must be the same length, got "
            f"{len(timestamps)} and {len(points)}"
        )
    if len(points) == 0:
        return []

    xs = [point.x if point is not None else None for point in points]
    ys = [point.y if point is not None else None for point in points]
    zs = [point.z if point is not None else None for point in points]

    velocity_x = _differentiate_axis(xs, timestamps, config.max_gap_s)
    velocity_y = _differentiate_axis(ys, timestamps, config.max_gap_s)
    velocity_z = _differentiate_axis(zs, timestamps, config.max_gap_s)

    accel_x = _differentiate_axis(velocity_x, timestamps, config.max_gap_s)
    accel_y = _differentiate_axis(velocity_y, timestamps, config.max_gap_s)
    accel_z = _differentiate_axis(velocity_z, timestamps, config.max_gap_s)

    jerk_x = _differentiate_axis(accel_x, timestamps, config.max_gap_s)
    jerk_y = _differentiate_axis(accel_y, timestamps, config.max_gap_s)
    jerk_z = _differentiate_axis(accel_z, timestamps, config.max_gap_s)

    samples: list[KinematicSample | None] = []
    for i, point in enumerate(points):
        if point is None:
            samples.append(None)
            continue
        samples.append(
            KinematicSample(
                position=point,
                velocity=_maybe_point(velocity_x[i], velocity_y[i], velocity_z[i]),
                acceleration=_maybe_point(accel_x[i], accel_y[i], accel_z[i]),
                jerk=_maybe_point(jerk_x[i], jerk_y[i], jerk_z[i]),
            )
        )
    return samples


def compute_keypoint_derivatives(
    sequence: PoseSequence,
    name: KeypointName,
    config: DerivativeConfig | None = None,
) -> list[KinematicSample | None]:
    """Velocity/acceleration/jerk trajectory for one named keypoint.

    Convenience adapter over `compute_point_derivatives` for the common
    case of analyzing a single joint (e.g. "how fast is the left wrist
    moving") directly from a PoseSequence.
    """
    timestamps = [frame.timestamp_s for frame in sequence]
    points = [_keypoint_to_point(frame.get(name)) for frame in sequence]
    return compute_point_derivatives(timestamps, points, config)


def compute_com_derivatives(
    sequence: PoseSequence,
    com_trajectory: Sequence[CenterOfMassEstimate | None],
    config: DerivativeConfig | None = None,
) -> list[KinematicSample | None]:
    """Velocity/acceleration/jerk trajectory for a center-of-mass trajectory.

    Convenience adapter over `compute_point_derivatives` for CoM
    trajectories produced by
    `climbiq.kinematics.com.compute_center_of_mass_trajectory`. `sequence`
    is only used for frame timestamps -- it must be the same PoseSequence
    (or at least the same length and timing) that `com_trajectory` was
    computed from.

    Raises:
        ValueError: if `sequence` and `com_trajectory` differ in length.
    """
    if len(sequence) != len(com_trajectory):
        raise ValueError(
            f"sequence and com_trajectory must be the same length, got "
            f"{len(sequence)} and {len(com_trajectory)}"
        )
    timestamps = [frame.timestamp_s for frame in sequence]
    points = [
        Point3D(estimate.x, estimate.y, estimate.z) if estimate is not None else None
        for estimate in com_trajectory
    ]
    return compute_point_derivatives(timestamps, points, config)


def _keypoint_to_point(keypoint: PoseKeypoint | None) -> Point3D | None:
    if keypoint is None:
        return None
    return Point3D(keypoint.x, keypoint.y, keypoint.z)


def _maybe_point(x: float | None, y: float | None, z: float | None) -> Point3D | None:
    if x is None or y is None:
        return None
    return Point3D(x, y, z)


def _differentiate_axis(
    values: Sequence[float | None],
    timestamps: Sequence[float],
    max_gap_s: float,
) -> list[float | None]:
    """Backward-difference derivative of one scalar axis, gap-aware.

    result[i] = (values[i] - values[j]) / (timestamps[i] - timestamps[j]),
    where j is the most recent earlier index with a non-None value and
    timestamps[i] - timestamps[j] <= max_gap_s. result[i] is None if
    values[i] itself is None, if no such j exists (start of sequence, or
    every earlier value was too long ago), or -- defensively -- if the
    computed dt is non-positive.
    """
    result: list[float | None] = [None] * len(values)
    last_valid_index: int | None = None

    for i, value in enumerate(values):
        if value is None:
            continue
        if last_valid_index is not None:
            dt = timestamps[i] - timestamps[last_valid_index]
            if 0 < dt <= max_gap_s:
                result[i] = (value - values[last_valid_index]) / dt  # type: ignore[operator]
        last_valid_index = i

    return result