# ClimbIQ

AI-powered climbing movement analysis. Turns a climbing video into
structured, explainable movement metrics -- pose extraction, kinematics,
and (eventually) a learned movement representation -- before any hold
detection or coaching logic gets involved.

## Philosophy

- **Climber-first, not wall-first.** Movement analysis is built and
  validated before hold detection is even started.
- **Modular by contract.** Every pose-estimation backend implements the
  same `PoseEstimator` interface and returns the same `PoseSequence`.
  Downstream code (kinematics, metrics, eventually a learned movement
  embedding) depends only on that contract, never on which model produced
  it. See `climbiq/pose/base.py` and `climbiq/pose/types.py`.
- **CLI-first.** No database, API, or app yet. The near-term goal is
  `python analyze.py climb.mp4` producing a real, trustworthy report.
  Productization (FastAPI, Postgres, Flutter) comes after the analysis
  engine is validated, not before.

## Project layout

```
climbiq/
    pose/
        types.py               # implementation-agnostic pose data structures
        base.py                 # PoseEstimator interface
        mediapipe_estimator.py    # MediaPipe-backed implementation
tests/
    pose/
        test_types.py
```

Planned next: `climbiq/kinematics/` (smoothing, center of mass, velocity/
jerk), `climbiq/metrics/` (phase segmentation, pause/smoothness/efficiency),
`climbiq/reports/` (JSON + PDF report generation).

## Setup

Requires [Poetry](https://python-poetry.org/) and Python 3.10-3.12
(MediaPipe does not yet support 3.13).

```bash
poetry install
```

### MediaPipe model file

`MediaPipePoseEstimator` requires a MediaPipe PoseLandmarker `.task` model
file, which is not bundled with the `mediapipe` package. Download one from
the [PoseLandmarker models page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models)
(`pose_landmarker_full.task` is a reasonable default) and keep it out of
git -- `*.task` is already in `.gitignore`.

```bash
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

## Development

```bash
poetry run pytest        # tests
poetry run ruff check .  # lint
poetry run mypy climbiq  # type-check
```
