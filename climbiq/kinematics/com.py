"""Center-of-mass estimation via a segmental anthropometric model.

Computes an approximate whole-body center of mass (CoM) for each frame by
treating the body as a set of rigid segments (upper arm, forearm, hand,
thigh, shank, foot, trunk, head/neck), each contributing a fraction of
total body mass located at a fixed proportion along that segment. The mass
fractions and segment CoM locations used here follow the widely-cited
anthropometric parameters in Winter, D.A. (2009), "Biomechanics and Motor
Control of Human Movement" -- a standard reference in sports biomechanics,
not an invented heuristic.

Important honesty about what this estimate actually is: it uses
population-average mass distribution for a generic adult, not this
specific climber's real body composition, and two segments (trunk,
head/neck) are approximated using the nearest available BlazePose
landmarks as substitutes for the anatomical reference points (C7 vertebra,
greater trochanter) the original anthropometric tables actually use.
Treat CoM output as a consistent, comparable *signal* -- useful for
tracking trends within one climber, comparing efficiency across sessions,
spotting excessive lateral sway -- not as a millimeter-accurate physical
measurement. Personalizing these fractions to an individual climber's
actual measurements is a natural extension once ClimbIQ's personalization
layer exists; the segment structure here doesn't need to change for that,
only the fraction values would move from population defaults to per-climber
values.

Responsibility of this module, and only this module: PoseFrame/PoseSequence
in, center-of-mass estimate(s) out. No velocity, no jerk, no phase
segmentation -- those live in sibling kinematics/metrics modules and
consume this module's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from climbiq.pose.types import KeypointName, PoseFrame, PoseKeypoint, PoseSequence


class Point3D(NamedTuple):
    """A plain 3D (or 2D, if z is unavailable) point, decoupled from any
    specific named landmark -- unlike PoseKeypoint, a center-of-mass point
    isn't any single joint, so it doesn't carry a KeypointName, visibility,
    or presence score."""

    x: float
    y: float
    z: float | None


@dataclass(frozen=True, slots=True)
class CenterOfMassEstimate:
    """Estimated whole-body center of mass for a single frame.

    Attributes:
        x: Horizontal coordinate, in the same coordinate space/units as
            the source PoseSequence (see CoordinateSpace).
        y: Vertical coordinate, same space as x.
        z: Depth coordinate, if every contributing segment had a z value;
            None otherwise (e.g. the source PoseSequence has no depth
            data at all, or a 2D-only backend produced this frame).
        mass_coverage: Fraction, in [0, 1], of the anthropometric model's
            total body mass that was actually included in this estimate.
            1.0 means every modeled segment had both its endpoint
            landmarks detected. Lower values mean some segments were
            skipped because a required landmark was missing (occlusion,
            out of frame, low-confidence detection never recorded) --
            the CoM is still computed from whatever was available, but
            with reduced reliability that this field makes explicit
            rather than hiding. Downstream code computing e.g. CoM
            velocity should treat low-coverage frames with suspicion,
            not equal trust to high-coverage ones.
    """

    x: float
    y: float
    z: float | None
    mass_coverage: float


# An anchor is either a single landmark (its raw position is used
# directly) or a pair of landmarks (their midpoint is used) -- the latter
# lets us build a segment endpoint like "trunk top" from the shoulder
# midpoint, since no single BlazePose landmark corresponds to it.
_Anchor = KeypointName | tuple[KeypointName, KeypointName]


@dataclass(frozen=True, slots=True)
class _Segment:
    """One rigid-body segment in the anthropometric model.

    Attributes:
        name: Human-readable identifier, for debugging/error messages only.
        proximal: Anchor at the segment's proximal (closer to torso) end.
        distal: Anchor at the segment's distal (farther from torso) end.
        mass_fraction: This segment's mass as a fraction of total body
            mass (Winter, 2009).
        com_fraction_from_proximal: Where along the proximal->distal line
            this segment's own center of mass sits, as a fraction in
            [0, 1] (0 = at the proximal anchor, 1 = at the distal anchor).
    """

    name: str
    proximal: _Anchor
    distal: _Anchor
    mass_fraction: float
    com_fraction_from_proximal: float


# fmt: off
_SEGMENTS: tuple[_Segment, ...] = (
    # Bilateral limb segments (each side's mass fraction and CoM-location
    # fraction from Winter, 2009). Hand's distal anchor uses the index
    # finger landmark as a stand-in for "hand tip" -- BlazePose has no
    # single "middle fingertip" landmark, and index is a close enough
    # proxy for this purpose.
    _Segment(
        "left_upper_arm", KeypointName.LEFT_SHOULDER, KeypointName.LEFT_ELBOW, 0.028, 0.436
    ),
    _Segment(
        "right_upper_arm", KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_ELBOW, 0.028, 0.436
    ),
    _Segment("left_forearm", KeypointName.LEFT_ELBOW, KeypointName.LEFT_WRIST, 0.016, 0.430),
    _Segment("right_forearm", KeypointName.RIGHT_ELBOW, KeypointName.RIGHT_WRIST, 0.016, 0.430),
    _Segment("left_hand", KeypointName.LEFT_WRIST, KeypointName.LEFT_INDEX, 0.006, 0.506),
    _Segment("right_hand", KeypointName.RIGHT_WRIST, KeypointName.RIGHT_INDEX, 0.006, 0.506),
    _Segment("left_thigh", KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE, 0.100, 0.433),
    _Segment("right_thigh", KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE, 0.100, 0.433),
    _Segment("left_shank", KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE, 0.045, 0.433),
    _Segment("right_shank", KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE, 0.045, 0.433),
    _Segment("left_foot", KeypointName.LEFT_HEEL, KeypointName.LEFT_FOOT_INDEX, 0.0145, 0.500),
    _Segment("right_foot", KeypointName.RIGHT_HEEL, KeypointName.RIGHT_FOOT_INDEX, 0.0145, 0.500),
    # Trunk: proximal anchor is the shoulder midpoint (approximating the
    # C7-vertebra reference point used in the original anthropometric
    # study), distal anchor is the hip midpoint (approximating the
    # greater-trochanter reference). CoM fraction of 0.5 is a
    # simplification -- treat as approximate, see module docstring.
    _Segment(
        "trunk",
        (KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER),
        (KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP),
        0.497,
        0.500,
    ),
    # Head+neck: proximal anchor is the shoulder midpoint (approximating
    # the neck base), distal anchor is the nose (approximating the head's
    # forward extent). Also a simplification -- see module docstring.
    _Segment(
        "head_neck",
        (KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER),
        KeypointName.NOSE,
        0.081,
        0.500,
    ),
)
# fmt: on

# Segment mass fractions sum to ~0.997, not exactly 1.0 -- consistent with
# Winter's published table (the remaining ~0.3% is rounding in the
# original anthropometric study). Used to normalize mass_coverage to a
# clean 0-1 range regardless of that rounding.
_TOTAL_MODEL_MASS: float = sum(segment.mass_fraction for segment in _SEGMENTS)


def _resolve_anchor(frame: PoseFrame, anchor: _Anchor) -> Point3D | None:
    """Resolve an anchor to a concrete point, or None if unavailable.

    A tuple anchor resolves to the midpoint of its two landmarks, and is
    only available if *both* landmarks were detected in this frame.
    """
    if isinstance(anchor, tuple):
        first = frame.get(anchor[0])
        second = frame.get(anchor[1])
        if first is None or second is None:
            return None
        return _midpoint(first, second)

    keypoint = frame.get(anchor)
    if keypoint is None:
        return None
    return Point3D(keypoint.x, keypoint.y, keypoint.z)


def _midpoint(a: PoseKeypoint, b: PoseKeypoint) -> Point3D:
    z = (a.z + b.z) / 2.0 if a.z is not None and b.z is not None else None
    return Point3D((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, z)


def _lerp(start: Point3D, end: Point3D, t: float) -> Point3D:
    z = start.z + (end.z - start.z) * t if start.z is not None and end.z is not None else None
    return Point3D(
        start.x + (end.x - start.x) * t,
        start.y + (end.y - start.y) * t,
        z,
    )


def estimate_center_of_mass(frame: PoseFrame) -> CenterOfMassEstimate | None:
    """Estimate whole-body center of mass for a single frame.

    Returns None if the frame is empty (no person detected) or if none of
    the modeled segments could be resolved (every relevant landmark
    missing) -- there is no meaningful CoM to report in that case,
    distinct from a CoM estimate with low `mass_coverage`.

    Segments whose required landmarks aren't present in this frame are
    skipped, and their mass is excluded from the weighted average rather
    than assumed to be at some default location -- see
    `CenterOfMassEstimate.mass_coverage` for how to tell a fully-supported
    estimate from a partially-supported one.
    """
    if frame.is_empty:
        return None

    weighted_points: list[tuple[Point3D, float]] = []
    for segment in _SEGMENTS:
        proximal_point = _resolve_anchor(frame, segment.proximal)
        distal_point = _resolve_anchor(frame, segment.distal)
        if proximal_point is None or distal_point is None:
            continue
        segment_com = _lerp(proximal_point, distal_point, segment.com_fraction_from_proximal)
        weighted_points.append((segment_com, segment.mass_fraction))

    if not weighted_points:
        return None

    included_mass = sum(weight for _, weight in weighted_points)

    x = sum(point.x * weight for point, weight in weighted_points) / included_mass
    y = sum(point.y * weight for point, weight in weighted_points) / included_mass

    z_contributions = [
        (point.z, weight) for point, weight in weighted_points if point.z is not None
    ]
    z: float | None = None
    if len(z_contributions) == len(weighted_points):
        z_mass = sum(weight for _, weight in z_contributions)
        z = sum(value * weight for value, weight in z_contributions) / z_mass

    return CenterOfMassEstimate(
        x=x, y=y, z=z, mass_coverage=min(1.0, included_mass / _TOTAL_MODEL_MASS)
    )


def compute_center_of_mass_trajectory(
    sequence: PoseSequence,
) -> list[CenterOfMassEstimate | None]:
    """Estimate center of mass for every frame in a PoseSequence.

    Returns a list the same length as `sequence`, index-aligned with it
    (result[i] corresponds to sequence[i]) -- entries are None wherever
    `estimate_center_of_mass` couldn't produce an estimate, most commonly
    frames where `PoseFrame.is_empty` is True. This mirrors how missing
    detections flow through the rest of ClimbIQ: represented explicitly
    as None/empty, never silently dropped or interpolated by this layer.
    """
    return [estimate_center_of_mass(frame) for frame in sequence]