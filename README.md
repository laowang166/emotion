# Classroom Emotion PoC

Offline classroom face tracking and observable-state analysis for Intel PTL iGPU.
The project uses anonymous, session-local track IDs and does not perform identity
recognition. Raw or annotated video is not written by default.

## Current implementation

- Timestamp-based three-second emotion smoothing.
- Conservative classroom-state mapping with an explicit `unknown` state.
- Anonymous JSON record contract and JSONL writer.
- OpenVINO device and capability probe.
- Configuration and model provenance manifests.
- SCRFD anchor, box, landmark, and NMS decoding.
- ByteTrack adapter that advances on every video frame.
- Static OpenVINO batch planning for classroom-scale face counts.
- Pluggable face-detection and emotion-classification backends (OpenVINO or synthetic).
- End-to-end video analysis pipeline with optional annotated-video output.
- Synthetic detector, classifier, and sample-video generator for weight-free validation.

Real SCRFD and HSEmotion weights are not bundled. The OpenVINO backends are wired
and ready, but model files are never downloaded implicitly; the synthetic backends
let the full pipeline run and be validated without them.

## Usage

Probe the iGPU:

```powershell
classroom-emotion probe --device GPU
```

Generate a synthetic validation video and analyze it without any model weights:

```powershell
classroom-emotion make-sample --output samples/classroom_sample.mp4 --seconds 4 --faces 6
classroom-emotion analyze --input samples/classroom_sample.mp4 --output outputs/predictions.jsonl --annotate outputs/annotated.mp4 --synthetic
```

Analyze a real video once SCRFD and emotion weights are available:

```powershell
classroom-emotion analyze --input classroom.mp4 --output outputs/predictions.jsonl --detector-model models/weights/scrfd_2.5g.xml --emotion-model models/weights/hsemotion.xml --device GPU
```

## Development

Requires Python 3.11 through 3.13. The local development environment uses Python 3.12
because the system-default Python 3.14 is outside the supported inference stack.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

After installing a PTL-compatible OpenVINO release, inspect the iGPU:

```powershell
classroom-emotion probe --device GPU
```

Model provenance and license review status live in `models/manifest.yaml`. Do not
treat repository source-code licenses as permission to use all pretrained weights or
training datasets.