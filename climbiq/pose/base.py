"""Abstract interface for pose estimation backends.

This module defines the *only* contract the rest of ClimbIQ depends on for
turning a video into pose data. It must never import or reference a
specific pose-estimation library (MediaPipe, YOLO-Pose, HRNet, ...) --
those live in sibling modules (e.g. `mediapipe_estimator.py`) that
implement this interface. If this file ever imports `mediapipe`, `torch`,
or `cv2`, that's a sign the abstraction has leaked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from climbiq.pose.types import PoseSequence


class PoseEstimator(ABC):
    """Contract for any component that turns a video into a PoseSequence.

    Implementations are responsible for:
      - Managing their own model lifecycle internally (loading weights,
        opening/closing any native resources).
      - Mapping their backend-specific landmark output onto the canonical
        `KeypointName` vocabulary defined in `pose.types`.
      - Declaring which `CoordinateSpace` their output coordinates use.
      - Doing exactly this, and nothing else. No smoothing, no kinematics,
        no metric computation, no report generation -- each of those is a
        separate module with its own single responsibility, operating on
        the `PoseSequence` this class produces.

    Single-responsibility is enforced here deliberately: the moment an
    estimator implementation starts doing its own smoothing "because it's
    convenient", downstream modules can no longer assume raw, unsmoothed
    pose data, and the pipeline's stages stop being independently
    swappable and testable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier for this estimator implementation.

        Used only for provenance (stored in `PoseSequence.estimator_name`)
        and logging. Must never be branched on by downstream logic --
        doing so would reintroduce a dependency on backend identity that
        this interface exists to remove. Example:
        `"mediapipe_pose_landmarker[heavy]"`.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate(self, video_path: str) -> PoseSequence:
        """Run pose estimation over an entire video and return the result.

        Args:
            video_path: Path to a video file readable by this estimator.

        Returns:
            A PoseSequence covering every frame of the video, in order,
            with no gaps -- see `PoseSequence.frames` for the no-gaps
            invariant that downstream finite-difference kinematics depend
            on. Frames where no person was detected must still be present,
            represented as a `PoseFrame` with empty keypoints.

        Raises:
            FileNotFoundError: if `video_path` does not exist.
            ValueError: if the video cannot be decoded, reports invalid
                metadata (e.g. zero or negative fps), or contains zero
                frames.
        """
        raise NotImplementedError
