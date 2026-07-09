"""PoseEstimator implementation backed by MediaPipe's PoseLandmarker task.

Responsibility of this module, and only this module: video in, PoseSequence
out. No smoothing, no kinematics, no metric computation, no report
generation -- those all operate on the PoseSequence this module returns,
and none of them need to know MediaPipe was ever involved. This is the
*only* file in ClimbIQ allowed to import `mediapipe`.

Setup note: MediaPipe's PoseLandmarker requires a separately downloaded
`.task` model file (it is not bundled with the `mediapipe` pip package).
Download one from
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
-- `pose_landmarker_full.task` is a reasonable default -- and pass its path
to `MediaPipePoseEstimator(model_asset_path=...)`.
"""

from __future__ import annotations

import os

import cv2
import mediapipe as mp

from climbiq.pose.base import PoseEstimator
from climbiq.pose.types import (
    CoordinateSpace,
    KeypointName,
    PoseFrame,
    PoseKeypoint,
    PoseSequence,
    VideoMetadata,
)

# MediaPipe's BlazePose model always emits its 33 landmarks in this fixed
# order. This tuple is the ENTIRE point of contact between "MediaPipe's
# specific output format" and "ClimbIQ's canonical KeypointName vocabulary"
# -- it lives here, in the one file allowed to know MediaPipe exists, and
# nowhere else in the codebase.
_MEDIAPIPE_LANDMARK_ORDER: tuple[KeypointName, ...] = (
    KeypointName.NOSE,
    KeypointName.LEFT_EYE_INNER,
    KeypointName.LEFT_EYE,
    KeypointName.LEFT_EYE_OUTER,
    KeypointName.RIGHT_EYE_INNER,
    KeypointName.RIGHT_EYE,
    KeypointName.RIGHT_EYE_OUTER,
    KeypointName.LEFT_EAR,
    KeypointName.RIGHT_EAR,
    KeypointName.MOUTH_LEFT,
    KeypointName.MOUTH_RIGHT,
    KeypointName.LEFT_SHOULDER,
    KeypointName.RIGHT_SHOULDER,
    KeypointName.LEFT_ELBOW,
    KeypointName.RIGHT_ELBOW,
    KeypointName.LEFT_WRIST,
    KeypointName.RIGHT_WRIST,
    KeypointName.LEFT_PINKY,
    KeypointName.RIGHT_PINKY,
    KeypointName.LEFT_INDEX,
    KeypointName.RIGHT_INDEX,
    KeypointName.LEFT_THUMB,
    KeypointName.RIGHT_THUMB,
    KeypointName.LEFT_HIP,
    KeypointName.RIGHT_HIP,
    KeypointName.LEFT_KNEE,
    KeypointName.RIGHT_KNEE,
    KeypointName.LEFT_ANKLE,
    KeypointName.RIGHT_ANKLE,
    KeypointName.LEFT_HEEL,
    KeypointName.RIGHT_HEEL,
    KeypointName.LEFT_FOOT_INDEX,
    KeypointName.RIGHT_FOOT_INDEX,
)

_SUPPORTED_OUTPUT_SPACES = (
    CoordinateSpace.WORLD_METERS,
    CoordinateSpace.NORMALIZED_IMAGE,
    CoordinateSpace.PIXEL,
)


def _clamp01(value: float | None) -> float | None:
    """Clamp a MediaPipe confidence score into [0, 1].

    MediaPipe occasionally emits values that overshoot [0, 1] by a tiny
    floating-point margin (e.g. 1.0000001). PoseKeypoint validates its
    visibility/presence fields strictly, so this guards against a spurious
    ValueError caused by float noise rather than a real data problem.
    """
    if value is None:
        return None
    return min(1.0, max(0.0, float(value)))


class MediaPipePoseEstimator(PoseEstimator):
    """Extracts pose via MediaPipe's PoseLandmarker task, in VIDEO mode.

    Single-climber assumption: `num_poses` defaults to 1 and only the
    first detected person (MediaPipe's own top-confidence ordering) is
    kept per frame. This matches ClimbIQ's current scope -- one climber on
    the wall -- and is called out explicitly here as a real assumption,
    not treated as free multi-person support. Revisit if bouldering
    partner-check videos or multi-climber footage become in-scope.
    """

    def __init__(
        self,
        model_asset_path: str,
        *,
        output_space: CoordinateSpace = CoordinateSpace.WORLD_METERS,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Configure a MediaPipe-backed PoseEstimator.

        Args:
            model_asset_path: Path to a downloaded MediaPipe PoseLandmarker
                `.task` model file. See module docstring for where to get one.
            output_space: Which coordinate space the returned PoseSequence
                should use. WORLD_METERS (the default) is preferred for any
                kinematic analysis, since it's a metric space independent
                of camera distance and video resolution -- see
                `CoordinateSpace` for the full tradeoff. PIXEL requires the
                source video to report a valid width/height.
            num_poses: Maximum number of people MediaPipe should detect
                per frame. ClimbIQ currently only consumes the first
                detected person regardless of this value -- see class
                docstring.
            min_pose_detection_confidence: Passed through to MediaPipe;
                see MediaPipe's PoseLandmarkerOptions documentation.
            min_pose_presence_confidence: Passed through to MediaPipe.
            min_tracking_confidence: Passed through to MediaPipe.

        Raises:
            FileNotFoundError: if `model_asset_path` does not exist.
            ValueError: if `output_space` is not supported by this backend.
        """
        if output_space not in _SUPPORTED_OUTPUT_SPACES:
            raise ValueError(
                f"MediaPipePoseEstimator does not support output_space="
                f"{output_space!r}; supported values are "
                f"{[s.value for s in _SUPPORTED_OUTPUT_SPACES]}"
            )
        if not os.path.isfile(model_asset_path):
            raise FileNotFoundError(
                f"MediaPipe pose landmarker model not found at "
                f"{model_asset_path!r}. Download a .task model from "
                "https://ai.google.dev/edge/mediapipe/solutions/vision/"
                "pose_landmarker#models (e.g. pose_landmarker_full.task) "
                "and pass its path here."
            )

        self._model_asset_path = model_asset_path
        self._output_space = output_space
        self._num_poses = num_poses
        self._min_pose_detection_confidence = min_pose_detection_confidence
        self._min_pose_presence_confidence = min_pose_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence

    @property
    def name(self) -> str:
        model_name = os.path.splitext(os.path.basename(self._model_asset_path))[0]
        return f"mediapipe_pose_landmarker[{model_name}]"

    def estimate(self, video_path: str) -> PoseSequence:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path!r}")

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video for reading: {video_path!r}")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 0:
                raise ValueError(
                    f"Video {video_path!r} reports an invalid fps ({fps}); "
                    "cannot compute frame timestamps without it."
                )
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=self._model_asset_path
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=self._num_poses,
                min_pose_detection_confidence=self._min_pose_detection_confidence,
                min_pose_presence_confidence=self._min_pose_presence_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )

            frames: list[PoseFrame] = []
            frame_index = 0

            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    read_ok, bgr_frame = capture.read()
                    if not read_ok:
                        break

                    timestamp_s = frame_index / fps
                    # detect_for_video requires strictly increasing,
                    # integer millisecond timestamps.
                    timestamp_ms = int(round(timestamp_s * 1000))

                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    keypoints = self._extract_keypoints(result, width, height)

                    frames.append(
                        PoseFrame(
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                            keypoints=keypoints,
                        )
                    )
                    frame_index += 1

            if frame_index == 0:
                raise ValueError(f"Video {video_path!r} contains zero readable frames.")

            metadata = VideoMetadata(
                fps=fps,
                width=width,
                height=height,
                duration_s=frame_index / fps,
                source_path=video_path,
            )
            return PoseSequence(
                frames=frames,
                coordinate_space=self._output_space,
                metadata=metadata,
                estimator_name=self.name,
            )
        finally:
            capture.release()

    def _extract_keypoints(
        self,
        result: mp.tasks.vision.PoseLandmarkerResult,
        width: int | None,
        height: int | None,
    ) -> dict[KeypointName, PoseKeypoint]:
        """Map one frame's MediaPipe result onto canonical KeypointNames.

        Returns an empty dict (never raises) when MediaPipe detects no
        person in this frame -- an empty-but-present PoseFrame is the
        correct representation, not a missing frame. See
        `PoseFrame.is_empty`.
        """
        if self._output_space == CoordinateSpace.WORLD_METERS:
            landmark_lists = result.pose_world_landmarks
        else:
            landmark_lists = result.pose_landmarks

        if not landmark_lists:
            return {}

        # Single-climber assumption: keep only the first detected person.
        landmarks = landmark_lists[0]

        keypoints: dict[KeypointName, PoseKeypoint] = {}
        for keypoint_name, landmark in zip(_MEDIAPIPE_LANDMARK_ORDER, landmarks, strict=False):
            x, y = landmark.x, landmark.y
            if self._output_space == CoordinateSpace.PIXEL:
                if width is None or height is None:
                    raise ValueError(
                        "output_space=PIXEL requires known video width/"
                        "height, which could not be read from the source "
                        "video."
                    )
                x, y = x * width, y * height

            keypoints[keypoint_name] = PoseKeypoint(
                name=keypoint_name,
                x=x,
                y=y,
                z=getattr(landmark, "z", None),
                visibility=_clamp01(getattr(landmark, "visibility", None)),
                presence=_clamp01(getattr(landmark, "presence", None)),
            )
        return keypoints
