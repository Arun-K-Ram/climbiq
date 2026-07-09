"""Implementation-agnostic data structures for representing human pose over time.

These types define the *contract* between any pose-estimation backend
(MediaPipe, YOLO-Pose, HRNet, a custom model, ...) and every downstream
module in ClimbIQ: smoothing, kinematics, movement representation, metrics,
and reporting.

Design principle: nothing in this module may import or reference a specific
pose-estimation library. If you find yourself wanting to import mediapipe
(or torch, or cv2) here, that dependency belongs in a backend-specific
estimator module instead (e.g. climbiq/pose/mediapipe_estimator.py), which
is responsible for mapping that library's native output onto these types.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class KeypointName(str, Enum):
    """Canonical vocabulary of body landmarks.

    Why a canonical enum instead of raw strings or per-model indices?

    Every pose-estimation model has its own landmark taxonomy: MediaPipe
    BlazePose emits 33 points, COCO-based models (YOLO-Pose, HRNet-COCO)
    emit 17, OpenPose yet another set. If downstream code indexed keypoints
    by a model-specific integer (e.g. `keypoints[15]` happening to be
    MediaPipe's left wrist), swapping the pose backend would silently break
    every consumer that assumed that indexing.

    Instead, every `PoseEstimator` implementation is responsible for mapping
    its native output onto this shared vocabulary. Downstream code always
    asks for `KeypointName.LEFT_WRIST`, never a raw index. A model that
    doesn't localize a given point (e.g. a COCO-17 model has no per-finger
    landmarks) simply omits that key from `PoseFrame.keypoints` -- see
    `PoseFrame.get()`.

    The vocabulary below is a superset based on MediaPipe BlazePose's 33
    landmarks, chosen because it is the most complete widely-used topology
    among current pose backends and a strict superset of COCO's 17 points
    (so a COCO-based estimator maps cleanly onto a subset of this enum).
    Extending this enum for a future estimator that localizes points no
    current backend provides is a purely additive, backwards-compatible
    change -- nothing that already reads this enum needs to change.
    """

    NOSE = "nose"
    LEFT_EYE_INNER = "left_eye_inner"
    LEFT_EYE = "left_eye"
    LEFT_EYE_OUTER = "left_eye_outer"
    RIGHT_EYE_INNER = "right_eye_inner"
    RIGHT_EYE = "right_eye"
    RIGHT_EYE_OUTER = "right_eye_outer"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    MOUTH_LEFT = "mouth_left"
    MOUTH_RIGHT = "mouth_right"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_PINKY = "left_pinky"
    RIGHT_PINKY = "right_pinky"
    LEFT_INDEX = "left_index"
    RIGHT_INDEX = "right_index"
    LEFT_THUMB = "left_thumb"
    RIGHT_THUMB = "right_thumb"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"


class CoordinateSpace(str, Enum):
    """The coordinate system that x, y, z values in a PoseSequence are expressed in.

    This is sequence-level metadata, not per-keypoint, because a single
    estimator run produces coordinates in one consistent space throughout.
    Downstream code (kinematics, especially) MUST check this before doing
    arithmetic on coordinates: a velocity computed on NORMALIZED_IMAGE
    coordinates is "fraction of image width per second", which is not a
    physical speed and isn't comparable across videos of different
    resolution or camera distance, whereas a velocity computed on
    WORLD_METERS is.

    NORMALIZED_IMAGE: x, y in [0, 1], relative to image width/height,
        origin top-left. This is MediaPipe's default `pose_landmarks`
        output. z, if present, is on the same normalized scale as x,
        roughly proportional to depth relative to the hips -- treat it
        as a relative, unitless signal, not a metric distance.
    PIXEL: x, y in absolute pixel coordinates for the source frame's
        resolution, origin top-left. Useful for overlaying keypoints on
        the original video frame for visualization/debugging.
    WORLD_METERS: x, y, z in real-world metric units (approximately
        meters), origin at the subject's hip midpoint, independent of
        camera distance or image resolution. This is MediaPipe's
        `pose_world_landmarks` output. Preferred for any kinematic
        computation (velocity, jerk, center of mass) that should be
        comparable across videos shot at different distances or
        resolutions.
    """

    NORMALIZED_IMAGE = "normalized_image"
    PIXEL = "pixel"
    WORLD_METERS = "world_meters"


def _validate_unit_interval(value: float | None, field_name: str, owner_repr: str) -> None:
    if value is not None and not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{field_name} must be in [0, 1], got {value!r} for {owner_repr}"
        )


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    """A single detected landmark at a single point in time.

    Attributes:
        name: Canonical landmark identity (see KeypointName).
        x: Horizontal coordinate. Units/origin depend on the owning
            PoseSequence.coordinate_space.
        y: Vertical coordinate. Units/origin depend on coordinate_space.
        z: Depth coordinate, if the estimator provides one. None for
            estimators that are strictly 2D (most COCO-based detectors).
            Interpretation depends on coordinate_space -- see the
            CoordinateSpace docstring.
        visibility: Model's estimated probability, in [0, 1], that this
            landmark is visible (not occluded by another body part, an
            object, or the frame edge) right now. Not all estimators
            provide this; None if unavailable. Downstream code should
            treat a missing visibility as "unknown", never default it
            to 1.0 (that would silently hide occlusion, which is one of
            the most common failure modes in climbing footage).
        presence: Model's estimated probability, in [0, 1], that this
            landmark exists / was located at all, independent of whether
            it's currently occluded. Distinct from `visibility`: a
            landmark can be "present" (the model is confident about
            where the joint is) yet reported with low "visibility" (the
            model believes it's currently hidden and is extrapolating
            its position from context). Estimators that only expose a
            single combined confidence score should populate `presence`
            and leave `visibility` as None.

    Design note: this class intentionally has no coordinate_space field of
    its own. Storing it per-keypoint would allow a single frame to silently
    mix coordinate systems across landmarks, which must never happen -- the
    space is fixed once, for the whole PoseSequence, instead.
    """

    name: KeypointName
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None
    presence: float | None = None

    def __post_init__(self) -> None:
        owner_repr = f"keypoint {self.name!r}"
        _validate_unit_interval(self.visibility, "visibility", owner_repr)
        _validate_unit_interval(self.presence, "presence", owner_repr)

    @property
    def confidence(self) -> float | None:
        """Best-available single confidence score for this landmark.

        Convenience accessor for callers that don't need to distinguish
        visibility from presence and just want "how much should I trust
        this point". Prefers visibility (occlusion-aware) when available,
        since occlusion is the dominant failure mode in climbing footage
        (limbs crossing, chalk bag, camera angle behind a hold); falls
        back to presence; returns None if the estimator provided neither.
        """
        if self.visibility is not None:
            return self.visibility
        return self.presence


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """All detected landmarks for one climber at one point in time.

    Attributes:
        frame_index: Zero-based index of this frame within the source
            video, in decode order. Stable and unambiguous even for
            variable-frame-rate video; prefer this over timestamp_s for
            anything that needs exact frame alignment (e.g. re-reading
            the original frame for visualization).
        timestamp_s: Time of this frame in seconds from the start of the
            video. Derived from frame_index and the source video's fps at
            estimation time, but stored explicitly so downstream code
            never has to recompute it or carry fps around separately.
        keypoints: Mapping from canonical landmark name to its detection
            in this frame. Deliberately a Mapping, not a fixed-size
            structure indexed by every KeypointName: an estimator that
            only detects a subset of landmarks (e.g. a hands-only or
            COCO-17 model) simply omits the rest, and callers must handle
            that explicitly via `get()` rather than silently reading a
            placeholder/zero value for a joint that was never detected.
        detection_confidence: Optional frame-level confidence that a
            person was detected at all, as distinct from the per-landmark
            visibility/presence carried on each PoseKeypoint. None if the
            estimator doesn't expose one.

    A frame with an empty `keypoints` mapping is a valid, meaningful state
    -- "no person detected in this frame" -- and must be preserved rather
    than dropped, so frame_index/timestamp_s stay aligned with the source
    video and downstream finite-difference kinematics don't silently skip
    time. See `is_empty`.
    """

    frame_index: int
    timestamp_s: float
    keypoints: Mapping[KeypointName, PoseKeypoint]
    detection_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {self.frame_index}")
        if self.timestamp_s < 0:
            raise ValueError(f"timestamp_s must be >= 0, got {self.timestamp_s}")
        _validate_unit_interval(
            self.detection_confidence, "detection_confidence", f"frame {self.frame_index}"
        )
        # frozen=True only prevents *reassigning* self.keypoints; it does
        # nothing to stop mutation of the dict/object it points to. Close
        # that hole explicitly by coercing to a read-only mapping, so a
        # PoseFrame is actually immutable end-to-end, not just superficially.
        if not isinstance(self.keypoints, MappingProxyType):
            object.__setattr__(self, "keypoints", MappingProxyType(dict(self.keypoints)))

    def get(
        self, name: KeypointName, min_confidence: float | None = None
    ) -> PoseKeypoint | None:
        """Look up a landmark by name, with an optional confidence gate.

        Returns None if the landmark wasn't detected in this frame, or if
        it was detected but its `.confidence` is below `min_confidence`
        (when both are provided). This is the preferred access pattern
        over `frame.keypoints[name]`, since it turns "missing or
        unreliable" into an explicit None instead of a KeyError every
        caller has to remember to catch.
        """
        keypoint = self.keypoints.get(name)
        if keypoint is None:
            return None
        if min_confidence is not None:
            confidence = keypoint.confidence
            if confidence is not None and confidence < min_confidence:
                return None
        return keypoint

    def __contains__(self, name: KeypointName) -> bool:
        return name in self.keypoints

    @property
    def is_empty(self) -> bool:
        """True if no person/landmarks were detected in this frame."""
        return len(self.keypoints) == 0


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Descriptive metadata about the source video a PoseSequence was extracted from.

    Kept separate from PoseSequence itself so this bundle of "facts about
    the file" can be constructed and validated independently of the
    (potentially large) list of per-frame pose data.

    Attributes:
        fps: Frames per second of the source video, as reported by the
            decoder. Used to relate frame_index to timestamp_s.
        width: Source video frame width in pixels, if known. Required to
            convert NORMALIZED_IMAGE coordinates to PIXEL coordinates.
        height: Source video frame height in pixels, if known.
        duration_s: Total video duration in seconds, if known.
        source_path: Path or identifier of the source video, kept for
            provenance/debugging only. Not guaranteed to remain a valid
            filesystem path (the file may later move or be deleted) --
            treat as informational, never as a reliable handle back to
            the original file.
    """

    fps: float
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if self.width is not None and self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.height is not None and self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")


@dataclass(frozen=True, slots=True)
class PoseSequence:
    """A full, implementation-agnostic pose extraction result for one video.

    This is the single object every `PoseEstimator` implementation returns,
    and the single object every downstream module (smoothing, kinematics,
    movement representation, metrics, reporting) consumes. It is the
    contract of the entire system: swapping MediaPipe for YOLO-Pose or a
    custom model means writing a new PoseEstimator whose `estimate()`
    produces one of these -- nothing else in the codebase needs to change.

    Attributes:
        frames: Ordered sequence of PoseFrame, one per video frame, in
            strictly increasing frame_index order with no gaps (a frame
            where nobody was detected is still represented, as a PoseFrame
            with empty keypoints -- see PoseFrame.is_empty). This
            invariant matters: downstream code that computes velocities
            via finite differences between consecutive frames relies on
            frames being contiguous and evenly spaced in time.
        coordinate_space: The coordinate system every x/y/z value in
            `frames` is expressed in. See CoordinateSpace.
        metadata: Descriptive facts about the source video.
        estimator_name: Free-text identifier of the PoseEstimator
            implementation that produced this sequence (e.g.
            "mediapipe_pose_landmarker[heavy]"). Purely for provenance and
            debugging -- e.g. so a saved report can note which model
            produced the underlying data -- and must never be branched on
            by downstream logic. Doing so would silently reintroduce the
            backend coupling this whole module exists to prevent.
    """

    frames: Sequence[PoseFrame]
    coordinate_space: CoordinateSpace
    metadata: VideoMetadata
    estimator_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple):
            object.__setattr__(self, "frames", tuple(self.frames))
        for previous, current in zip(self.frames, self.frames[1:], strict=False):
            if current.frame_index <= previous.frame_index:
                raise ValueError(
                    "PoseSequence.frames must be strictly ordered by "
                    f"frame_index; got {previous.frame_index} followed by "
                    f"{current.frame_index}"
                )

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[PoseFrame]:
        return iter(self.frames)

    def __getitem__(self, index: int) -> PoseFrame:
        return self.frames[index]

    @property
    def duration_s(self) -> float:
        """Time span covered by this sequence, from first to last frame."""
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp_s - self.frames[0].timestamp_s

    @property
    def fps(self) -> float:
        return self.metadata.fps

    def frame_at_time(self, timestamp_s: float) -> PoseFrame:
        """Return the frame whose timestamp is closest to `timestamp_s`.

        Uses binary search since `frames` is time-ordered. Useful for
        mapping a movement-analysis result (e.g. "crux detected at 14.2s")
        back to a specific frame for report generation or visualization.

        Raises:
            IndexError: if this PoseSequence has no frames.
        """
        if not self.frames:
            raise IndexError("frame_at_time() called on an empty PoseSequence")
        timestamps = [frame.timestamp_s for frame in self.frames]
        i = bisect.bisect_left(timestamps, timestamp_s)
        if i == 0:
            return self.frames[0]
        if i == len(self.frames):
            return self.frames[-1]
        before, after = self.frames[i - 1], self.frames[i]
        if (timestamp_s - before.timestamp_s) <= (after.timestamp_s - timestamp_s):
            return before
        return after

    def detected_fraction(self) -> float:
        """Fraction of frames in which a person was detected at all.

        A cheap, useful data-quality signal worth surfacing early -- e.g.
        as a CLI warning -- before running expensive downstream analysis
        on footage where the estimator mostly failed to find the climber
        (common causes: camera too far away, climber leaves frame, heavy
        occlusion by the wall/holds).
        """
        if not self.frames:
            return 0.0
        detected = sum(1 for frame in self.frames if not frame.is_empty)
        return detected / len(self.frames)
