# Classroom Emotion PoC

Offline classroom face tracking and observable-state analysis for Intel PTL iGPU.
The project uses pseudonymous, session-local track IDs; an on-device face embedding
can reconnect the same face for up to 60 seconds within one run. It does not resolve
names or persist embeddings. Raw or annotated video is not written by default.

中文项目使用与配置说明见 [`docs/项目使用说明.md`](docs/项目使用说明.md)。

## Current implementation

- Timestamp-based three-second, quality-weighted emotion averaging (at least 4 valid observations).
- Classroom observation changes require a new candidate to remain stable for 1.2 seconds.
- Evidence-first teaching observations with an explicit insufficient-evidence result.
- Anonymous JSON record contract and JSONL writer.
- OpenVINO device and capability probe.
- Configuration and model provenance manifests.
- OpenVINO YuNet small-face detection with five landmarks and overlapping image tiles.
- Optional SCRFD decoding backend for deployments with approved weights.
- ByteTrack adapter that advances on every video frame.
- Conservative face-appearance re-identification across distant positions.
- Static OpenVINO batch planning for classroom-scale face counts.
- Pluggable face-detection and emotion-classification backends (OpenVINO or synthetic).
- End-to-end video analysis pipeline with optional annotated-video output.
- Synthetic detector, classifier, and sample-video generator for weight-free validation.

Model files are kept locally under the git-ignored `models/weights/` directory and
are never downloaded implicitly. YuNet is MIT-licensed; its model card reports faces
as small as approximately 10x10 pixels. HSEmotion source code is Apache-2.0, but
confirm the pretrained weights and VGGFace2 terms for the intended deployment.
The optional face action cues use MediaPipe Face Landmarker blendshapes for bilateral
eye-blink and brow-lowering candidates. Install the inference extra and obtain the
`face_landmarker.task` asset listed in `models/manifest.yaml`; all frame processing remains
local. These cues are not diagnoses: a blink is not proof of fatigue, and brow lowering is
not proof of confusion.

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

Download the YuNet, HSEmotion, and MediaPipe Face Landmarker files (PowerShell):

```powershell
New-Item -ItemType Directory -Force models/weights | Out-Null
curl.exe -fL https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -o models/weights/yunet.onnx
curl.exe -fL https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx -o models/weights/hsemotion.onnx
$reid = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP16"
curl.exe -fL "$reid/face-reidentification-retail-0095.xml" -o models/weights/face-reidentification-retail-0095.xml
curl.exe -fL "$reid/face-reidentification-retail-0095.bin" -o models/weights/face-reidentification-retail-0095.bin
$faceLandmarker = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
curl.exe -fL $faceLandmarker -o models/weights/face_landmarker.task
```

Check SHA-256 values for YuNet/HSEmotion and the XML SHA-384 against `models/manifest.yaml`.
The upstream FP16 BIN checksum is malformed (95 hex characters), so its SHA-384 is not
asserted here; verify the downloaded BIN against a trusted source before deployment. The
re-identification model is Apache-2.0;
confirm its model and training-data terms for the intended classroom deployment.

Convert the current HSEmotion ONNX weights to OpenVINO IR (this changes the deployment
format, not the architecture or its 224x224 input size):

```powershell
classroom-emotion export-openvino --model models/weights/hsemotion.onnx --output models/weights/hsemotion_openvino.xml --precision FP16
```

Use the exported XML explicitly with `--emotion-model models/weights/hsemotion_openvino.xml`.
The current weights remain `enet_b0_8_best_vgaf`; this export does not provide the
MobileNetV3 112x112 model.

Analyze a real video:

```powershell
classroom-emotion analyze --input classroom.mp4 --output outputs/predictions.jsonl --detector-model models/weights/yunet.onnx --emotion-model models/weights/hsemotion.onnx --device GPU
```

Analyze a live camera stream (press Ctrl+C to stop):

```powershell
classroom-emotion analyze --camera 0 --output outputs/live_predictions.jsonl --detector-model models/weights/yunet.onnx --emotion-model models/weights/hsemotion.onnx --device GPU
```

Evaluate emitted states against a human-labeled CSV:

```powershell
classroom-emotion evaluate --predictions outputs/predictions.jsonl --annotations outputs/annotations.csv
```

The CSV must contain `frame_index,track_id,state`, with one row per labeled observation.
New output uses `head_down_cue`, `confusion_cue`, or `unknown`; legacy state values remain
accepted for historical annotations.
Annotations must refer to the same analysis run because track IDs are session-local. Missing
predictions count as `unknown`; predictions without annotations are excluded from state metrics.
The report includes accuracy, per-state precision/recall/F1, macro-F1, and a confusion matrix.
This evaluates the state mapping only, not missed face detections, tracking identity switches,
or a student's actual internal mental state. Label frames using an independent human judgment
and retain only video and prediction data needed for the evaluation.

Open the local monitoring dashboard in a second terminal:

```powershell
python -m pip install -e ".[dashboard,inference,tracking]"
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Each browser session
uses that visitor's own browser camera and has an isolated tracker and JSONL output file. Frames
are sent to the host for inference; each active visitor creates a separate inference pipeline and
uses additional host CPU/GPU resources.

To let other devices on the LAN open the page, allow inbound TCP port 8501 in the host firewall.
In an elevated PowerShell, create a firewall rule:

```powershell
New-NetFirewallRule -DisplayName "Classroom dashboard" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8501
```

Share `https://<host-ip>:8501`. Browser camera access requires HTTPS from another device; plain
HTTP by IP may show the page but browsers will block camera permission. Configure a trusted
certificate whose subject includes the host IP. For example, with `mkcert` installed, create a
certificate for the LAN IP and use its generated certificate and key:

```powershell
$hostIp = "192.168.1.20"
mkcert -install
mkcert $hostIp
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501 --server.sslCertFile "$hostIp.pem" --server.sslKeyFile "$hostIp-key.pem"
```

The generated local CA must also be trusted by each client device, or use an organization-managed
certificate; otherwise browsers will reject the camera API. Replace the example IP with the host's
LAN IP. This dashboard has no user authentication, so restrict it to a trusted network and do not
forward the port to the public internet.

The WebRTC stream uses a STUN server to negotiate connectivity. Networks that block direct WebRTC
traffic also require a TURN relay configured in the `rtc_configuration` passed to
`webrtc_streamer`. Only expose the dashboard to trusted network users.

Video frames are processed in memory and are not saved. State records and emotion probabilities
are written to the selected JSONL path for the dashboard and later evaluation.
Each JSONL record also includes `primary_state`, `secondary_state`, `visual_evidence`,
`cognitive_load`, and `teacher_advice`. Emotion probabilities are smoothed over a configurable
three-second sliding window before mapping. The highest-scoring emotion label alone is mapped:
happy/surprise to an engagement candidate, neutral to an attentive candidate, explicit `confused`
to a confusion candidate, and sad/disgust to a bored/drowsy candidate. Brow-lowering cues map to
confusion, while low-head posture and closed-eye cues are evaluated separately. Emotion labels are
never added or combined across categories, and the output evidence names the selected label. Five-point face
landmarks also provide approximate Pitch/Yaw/Roll through solvePnP and estimated camera intrinsics;
these angles are coarse head-direction cues, not calibrated gaze or phone-use detection. Sustained
negative Pitch is reported as a head-down posture candidate. Combined with sustained eye closure,
it becomes a drowsiness/rest candidate; neither condition proves phone use or sleep.
For a new track, the preview counts down `3`, `2`, `1` until the window spans three seconds and
contains at least four valid inference observations (the default). Since emotion inference is
frame-throttled, readiness uses timestamps rather than a fixed count of camera frames. The first
window-backed result is emitted as soon as the window is ready; later state changes must remain
stable for the configured transition period.
These are candidates for teacher verification, not proof of a student's internal state. HSEmotion
does not emit a `confused` class; YuNet's five landmarks cannot measure eye closure or brow
movement. The local Face Landmarker task estimates bilateral blink and brow-lowering blendshapes;
a blink or lowered brow still does not prove fatigue or confusion. Unsupported evidence remains
`证据不足`. After a complete window, the strongest mapped class is shown even below its reference
confidence threshold, marked as a low-confidence candidate for verification. Cognitive load is
always `无法判断` with current inputs.

### Calibrate cues from live video

The dashboard's live-camera panel includes a cue-calibration section. Start the camera, select the
anonymous track to calibrate, collect a natural baseline for about 10 seconds, then select each of
皱眉, 闭眼, and 低头 while demonstrating that cue for about 3 seconds. Switch back to baseline
between demonstrations and repeat each cue 3-5 times. Calibration collects per-observation scores
in memory; it does not save frames, crops, or identities. Review the baseline P95 and action P10
values before saving. A cue is saved only when its baseline and action distributions separate and
its baseline noise duration is shorter than its action duration. Otherwise, collect more examples
or improve lighting and face visibility; no threshold is guessed for overlapping data.

Saved settings are written to the local, git-ignored `configs/cue_calibration.yaml` sidecar and are
loaded by the dashboard and the CLI when using `configs/default.yaml`. Stop and restart the camera
after saving to apply changes to a running session. Thresholds and persistence durations are
individual for brow lowering, bilateral eye closure, and head-down cues. These settings calibrate
observable signals only; they do not validate a student's attention, understanding, or fatigue.

The camera panel can select YuNet (default) or SCRFD-2.5G. SCRFD requires an independently
licensed ONNX weight file at `models/weights/scrfd_2.5g.onnx`; it is not bundled or
downloaded automatically. For CLI analysis, set `detection.model: scrfd_2_5g` in a
YAML config passed with `--config`, and pass the matching `--detector-model` path.
The classifier currently uses the verified 224x224 HSEmotion ONNX, not a 112x112
MobileNetV3 or DAN model. Do not change `emotion.input_size` without compatible weights.

An optional **JSON endpoint** in the camera panel, or `--json-endpoint URL` with `analyze`,
POSTs anonymous records as `application/json` while still writing local JSONL. Remote
endpoints require HTTPS; HTTP is restricted to localhost. Delivery is best-effort with a
bounded queue and a two-second network timeout: slow/offline servers can lose records,
so use the local JSONL for authoritative results. The CLI reports delivered/dropped
counts and the last delivery error. Do not enable remote delivery without appropriate
consent and retention controls for classroom data.

Face embeddings are computed locally from aligned five-point face crops and held only in
RAM. The gallery is bounded to eight samples per track and preserves the first template plus
recent views. Browser camera sessions retain IDs and their gallery until that browser stops the
camera; CLI/video analysis uses the configured
60-second identity retention. All in-memory identity features are cleared when analysis stops.
Embeddings and face crops are not written to JSONL or sent to the optional endpoint. IDs are
pseudonymous, not verified names; low-confidence or ambiguous matches receive a new ID instead.
If a later high-confidence match merges a provisional ID into an earlier ID, the local JSONL
history is rewritten to use the canonical ID. Records already sent to a remote endpoint and
frames already written to an annotated video cannot be retroactively changed.
Continuity does not survive camera restarts or app restarts, and appearance changes or severe
occlusion can still cause ID changes.

For a bounded PTL measurement, use a representative recorded classroom video:

```powershell
classroom-emotion analyze --input classroom.mp4 --output outputs/benchmark.jsonl --max-frames 300 --detector-model models/weights/yunet.onnx --emotion-model models/weights/hsemotion.onnx --device GPU
```

The report includes actual input resolution, measured processing FPS, and average
detection/tracking/emotion stage times. FPS from a video file measures processing throughput,
not camera-to-display latency; test detection recall, ID switches, and latency on the
intended camera and PTL system separately. A local Arc 140T GPU smoke test classified
50 blank crops using a 64-slot 224x224 batch in about 132 ms after warmup (initial
compile around 6 s); this is not a full-pipeline benchmark and does **not** support
the proposed 12 ms classifier or 40 FPS target. Tiling requires four detector passes
per detection frame, so the 23 ms total estimate must not be treated as measured.

Classroom observations are candidate visible cues, not measures of actual attention or mental
state. The five-landmark heuristic only approximates a low head angle; the added Face Landmarker
estimates eye-blink and brow-lowering blendshapes, but neither provides gaze direction or body
posture. The standard expression model has no `confused` class. Focus, distraction, frustration,
curiosity, anxiety, eureka, and cognitive load cannot be reliably assessed by these cues and should
be treated as insufficient evidence. A low-head/open-eyes/neutral combination may be shown as a
possible phone-use cue, but there is no phone or hand detector; looking at a book or writing can
produce the same cue. Sustained eye closure is a drowsiness cue, not a sleep detector.

Camera resolution and requested frame rate come from `configs/default.yaml`. The default
2x2 overlapping detector tiling increases the effective face scale for distant students at
the cost of four detector passes per detection frame. The camera may negotiate a lower
resolution or frame rate; use the reported stage latency and processing FPS on the target
PTL system before treating a real-time target as met. The browser camera requests the configured
resolution and displays the actual received frame dimensions; physical camera placement still
needs to be verified at installation.

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