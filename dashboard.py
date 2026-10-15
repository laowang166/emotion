"""Local Streamlit dashboard for monitoring anonymous JSONL analysis output."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from classroom_emotion.config import load_config

CAMERA_INPUT = load_config("configs/default.yaml").input
OPENVINO_EMOTION_MODEL = Path("models/weights/hsemotion_openvino.xml")
DEFAULT_EMOTION_MODEL = (
    str(OPENVINO_EMOTION_MODEL)
    if OPENVINO_EMOTION_MODEL.is_file()
    and OPENVINO_EMOTION_MODEL.with_suffix(".bin").is_file()
    else "models/weights/hsemotion.onnx"
)

TEXT = {
    "zh": {
        "sidebar_eyebrow": "本机监控",
        "sidebar_title": "数据源",
        "language": "语言 / Language",
        "path": "JSONL 结果文件",
        "path_help": "填写 analyze 命令中 --output 指定的路径。相对路径以项目根目录为准。",
        "refresh": "自动刷新",
        "refresh_option": "每 {seconds} 秒",
        "privacy": "启动后由此浏览器访问本机摄像头；画面发送到服务器推理，不保存原始视频。",
        "eyebrow": "课堂 / 状态观察",
        "title": "课堂状态看板",
        "subtitle": "匿名轨迹与时序状态 · 本机 JSONL 监控",
        "mapping_title": "线索说明与判断边界",
        "mapping_rows": [
            "系统输出面部表情模型概率和有限的面部/头部线索，不直接测量学习状态。",
            "候选观察是启发式结果，需结合课堂任务、互动或作答核实。",
            "系统不能据此判断是否听懂、专注、认知负荷或实际使用手机；证据不足时保留“证据不足”。",
        ],
        "mapping_temporal": (
            "时序规则：3 秒滑动窗口对情绪概率取均值，至少 4 条有效观测；"
            "候选观察需连续稳定 1.2 秒才切换。"
        ),
        "recent_write": "近期有写入",
        "no_recent_write": "历史结果 · 暂无新写入",
        "updated": "最近更新",
        "missing_file": "尚未找到结果文件：{path}。请先运行 analyze 命令并确认 --output 路径。",
        "read_error": "无法读取结果文件：{error}",
        "empty_file": "结果文件已创建，等待可用的人脸观测记录。",
        "tracks": "文件轨迹数",
        "observations": "文件观测数",
        "video_time": "视频时间",
        "latest_frame": "最新帧",
        "seconds": "秒",
        "skipped": "跳过了 {count} 行无效或尚未写完的 JSONL 记录。",
        "state_distribution": "当前教学观察分布",
        "track_states": "各轨迹最近观察",
        "track_filter": "筛选轨迹",
        "track_filter_all": "全部",
        "track_filter_candidate": "候选观察",
        "track_filter_unknown": "证据不足",
        "no_tracks_for_filter": "当前筛选条件下没有轨迹。",
        "track_id": "轨迹 ID",
        "state": "候选观察",
        "visual_evidence": "线索说明",
        "secondary_state": "次要观察",
        "no_secondary_state": "无",
        "unavailable_load": "无法判断",
        "cognitive_load": "认知负荷",
        "teacher_advice": "教学支持建议",
        "assessment_title": "教学支持提示",
        "observation_count": "观测次数",
        "eye_closed_cue": "闭眼线索",
        "frown_cue": "皱眉线索",
        "head_pose_angles": "头部姿态角（Pitch/Yaw/Roll）",
        "facial_cues_title": "面部动作线索",
        "cue_unavailable": "未测得",
        "cue_triggered": "候选触发",
        "cue_below_threshold": "未达阈值",
        "timestamp": "时间（秒）",
        "emotion_trend": "面部表情模型概率（不代表学习状态）",
        "recent_observations": "最近观测",
        "frame": "帧",
        "unknown": "未分类",
        "disclaimer": (
            "本看板只报告可见线索和面部表情模型输出，不代表学生真实心理或学习状态。"
            "头部姿态角是基于五点关键点和近似相机参数的粗略估计，不代表视线、手机使用或心理状态；"
            "眼睑/眉部 blendshape 只是候选线索。"
            "认知负荷无法据此判断；教学建议需结合课堂表现核实。"
            "轨迹 ID 只在本次运行中有效。"
        ),
        "source": "数据文件",
        "state_count": "轨迹数",
        "camera_title": "实时摄像头",
        "camera_index": "摄像头编号",
        "detector_type": "人脸检测器",
        "device": "推理设备",
        "detector_model": "人脸模型",
        "emotion_model": "情绪模型",
        "identity_model": "身份重识别模型",
        "identity_warning": (
            "人脸重识别模型 XML/BIN 未完整提供；ID 仅依赖短时轨迹关联，"
            "离开画面、遮挡或换位置后可能变化。"
        ),
        "identity_scope": (
            "ID 仅在当前浏览器的本次摄像头会话内有效；"
            "不同设备和新会话不会共享学生 ID。"
        ),
        "json_endpoint": "JSON 接收地址（可选）",
        "json_endpoint_help": "仅主动填写后才发送匿名记录。远程地址必须使用 HTTPS。",
        "delivery_stats": "服务端已接收请求 {sent} 条 · 丢弃 {dropped} 条 · 待发送 {pending} 条",
        "delivery_error": "最近发送失败：{error}。本地 JSONL 仍会继续写入。",
        "start_camera": "开启摄像头",
        "confirm_start": "确认覆盖并启动",
        "cancel_start": "取消启动",
        "stop_camera": "停止摄像头",
        "camera_preview": "实时画面",
        "camera_idle": "点击视频控件中的 START 并允许浏览器访问摄像头。远程访问必须使用 HTTPS。",
        "camera_waiting": "正在加载模型和连接摄像头。首次启动可能需要一些时间。",
        "camera_first_frame": "摄像头已连接，正在等待第一帧画面…",
        "camera_live_detail": "画面已到达，正在实时检测人脸并估计表情。",
        "camera_stopped_detail": "分析已结束。结果保存在 {path}，可在下方查看。",
        "camera_error_detail": "请检查摄像头权限、HTTPS、模型文件路径和推理设备后重试。",
        "confirm_output_warning": "再次点击将清空并覆盖 `{path}`；原始视频不会保存。",
        "calibration_title": "真实画面线索校准",
        "calibration_track": "校准轨迹",
        "calibration_phase": "采样阶段",
        "calibration_phase_off": "暂停采集",
        "calibration_phase_baseline": "自然基线",
        "calibration_phase_frown": "皱眉",
        "calibration_phase_eyes_closed": "闭眼",
        "calibration_phase_head_down": "低头",
        "calibration_instructions": (
            "先选自然基线并保持自然状态约 10 秒（可自然眨眼、轻微转头）；"
            "再切换到每种动作，保持约 3 秒后切回基线，重复 3-5 次。"
            "只采集本人主动演示的动作，不采集或保存原始视频。"
        ),
        "calibration_waiting_track": "摄像头运行并检测到人脸后，才能开始校准。",
        "calibration_counts": "当前已采集 {count} 条原始线索观测。",
        "calibration_threshold": "建议分数阈值",
        "calibration_duration": "建议最短持续时间（秒）",
        "calibration_baseline": "基线 P95",
        "calibration_target": "动作 P10",
        "calibration_samples": "基线/动作样本数",
        "calibration_warning": "尚无可靠建议：{warning}",
        "calibration_save": "保存可用校准结果",
        "calibration_saved": "已保存 {cues} 的校准值。停止并重新开启摄像头后生效。",
        "calibration_no_results": "目前没有可保存的线索；请补采缺少的阶段或改善画面条件。",
        "calibration_saved_cues": "皱眉、闭眼、低头",
        "camera_starting": "正在启动",
        "camera_running": "实时分析中",
        "camera_stopping": "正在停止",
        "camera_stopped": "已停止",
        "camera_error": "启动失败",
        "camera_stats": "已处理 {frames} 帧 · {fps:.1f} FPS · 实际输入 {resolution}",
        "camera_resolution_unknown": "分辨率待确认",
        "top_emotion": "主要表情（模型）",
        "confidence": "最高表情概率",
        "output_warning": "结果写入所选 JSONL 文件；原始视频不会保存。",
    },
    "en": {
        "sidebar_eyebrow": "LOCAL MONITOR",
        "sidebar_title": "Data source",
        "language": "Language / 语言",
        "path": "JSONL results file",
        "path_help": (
            "Use the path passed to --output in analyze. "
            "Relative paths use the project root."
        ),
        "refresh": "Auto refresh",
        "refresh_option": "Every {seconds}s",
        "privacy": (
            "The browser accesses its own camera after you start. Frames are sent to the "
            "server for inference; raw video is not saved."
        ),
        "eyebrow": "CLASSROOM / OBSERVATION",
        "title": "Classroom state monitor",
        "subtitle": "Anonymous tracks and temporal states · Local JSONL monitor",
        "mapping_title": "Cue interpretation limits",
        "mapping_rows": [
            "The system reports face-expression probabilities and limited facial/head cues; "
            "it does not measure learning state.",
            "Candidate observations are heuristic and need verification against classroom "
            "activity or student work.",
            "The system cannot determine comprehension, focus, cognitive load, or actual "
            "phone use; retain insufficient evidence when cues are inconclusive.",
        ],
        "mapping_temporal": (
            "Temporal rule: expression probabilities are averaged over a 3-second sliding "
            "window with at least 4 valid observations; a candidate must remain stable "
            "for 1.2 seconds before switching."
        ),
        "recent_write": "Recently updated",
        "no_recent_write": "Historical results · no recent updates",
        "updated": "Last update",
        "missing_file": "Results file not found: {path}. Run analyze and check its --output path.",
        "read_error": "Could not read results file: {error}",
        "empty_file": "Results file exists. Waiting for usable face observations.",
        "tracks": "Tracks in file",
        "observations": "Observations in file",
        "video_time": "Video time",
        "latest_frame": "Latest frame",
        "seconds": "s",
        "skipped": "Skipped {count} invalid or incomplete JSONL records.",
        "state_distribution": "Current teaching observations",
        "track_states": "Latest observation by track",
        "track_filter": "Filter tracks",
        "track_filter_all": "All",
        "track_filter_candidate": "Candidate observations",
        "track_filter_unknown": "Insufficient evidence",
        "no_tracks_for_filter": "No tracks match this filter.",
        "track_id": "Track ID",
        "state": "Candidate observation",
        "visual_evidence": "Cue summary",
        "secondary_state": "Secondary observation",
        "no_secondary_state": "None",
        "unavailable_load": "Unable to determine",
        "cognitive_load": "Cognitive load",
        "teacher_advice": "Teaching support suggestion",
        "assessment_title": "Teaching support prompt",
        "observation_count": "Observations",
        "eye_closed_cue": "Eye-closure cue",
        "frown_cue": "Frown cue",
        "head_pose_angles": "Head pose angles (Pitch/Yaw/Roll)",
        "facial_cues_title": "Facial action cues",
        "cue_unavailable": "Not measured",
        "cue_triggered": "Candidate triggered",
        "cue_below_threshold": "Below threshold",
        "timestamp": "Time (s)",
        "emotion_trend": "Face-expression model probabilities (not learning state)",
        "recent_observations": "Recent observations",
        "frame": "Frame",
        "unknown": "Unknown",
        "disclaimer": (
            "This dashboard reports visible cues and face-expression model outputs, "
            "not a student's actual psychological or learning state. "
            "Head-pose angles are coarse estimates from five landmarks and "
            "approximate camera intrinsics; they do not indicate gaze, phone use, "
            "or mental state. Eyelid/brow blendshapes are candidate cues. "
            "cognitive load cannot be determined. Verify suggestions against classroom work. "
            "Track IDs only last for this run."
        ),
        "source": "Source file",
        "state_count": "Tracks",
        "camera_title": "Live camera",
        "camera_index": "Camera index",
        "detector_type": "Face detector",
        "device": "Inference device",
        "detector_model": "Face model",
        "emotion_model": "Emotion model",
        "identity_model": "Face re-identification model",
        "identity_warning": (
            "The face re-identification XML/BIN files are missing; IDs rely on short-term tracking "
            "and may change after exits, occlusion, or position changes."
        ),
        "identity_scope": (
            "IDs are scoped to this browser's current camera session; devices and new sessions "
            "do not share student IDs."
        ),
        "json_endpoint": "JSON endpoint (optional)",
        "json_endpoint_help": (
            "Anonymous records are sent only when set. Remote URLs require HTTPS."
        ),
        "delivery_stats": "Sent {sent} · dropped {dropped} · queued {pending}",
        "delivery_error": "Latest delivery error: {error}. Local JSONL is still written.",
        "start_camera": "Start camera",
        "confirm_start": "Confirm overwrite and start",
        "cancel_start": "Cancel start",
        "stop_camera": "Stop camera",
        "camera_preview": "Live preview",
        "camera_idle": (
            "Click START in the video control and allow browser camera access. "
            "Remote access requires HTTPS."
        ),
        "camera_waiting": (
            "Loading models and connecting to the camera. "
            "First startup may take a while."
        ),
        "camera_first_frame": "Camera connected. Waiting for the first frame…",
        "camera_live_detail": "Frames received · detecting faces and estimating expressions live.",
        "camera_stopped_detail": "Analysis stopped. Results are in {path} and shown below.",
        "camera_error_detail": (
            "Check browser camera permission, HTTPS, model paths, and inference device, then retry."
        ),
        "confirm_output_warning": (
            "Click again to clear and replace `{path}`. Raw video is not saved."
        ),
        "calibration_title": "Calibrate cues from live video",
        "calibration_track": "Calibration track",
        "calibration_phase": "Sampling phase",
        "calibration_phase_off": "Pause collection",
        "calibration_phase_baseline": "Natural baseline",
        "calibration_phase_frown": "Frown",
        "calibration_phase_eyes_closed": "Eyes closed",
        "calibration_phase_head_down": "Head down",
        "calibration_instructions": (
            "Collect a natural baseline for about 10 seconds (blink and move naturally). "
            "For each cue, hold it for about 3 seconds, switch back to baseline, and repeat "
            "3-5 times. Only demonstrate your own actions; raw video is not saved."
        ),
        "calibration_waiting_track": (
            "Start the camera and wait for a detected face before calibrating."
        ),
        "calibration_counts": "Collected {count} raw cue observations.",
        "calibration_threshold": "Suggested score threshold",
        "calibration_duration": "Suggested minimum duration (s)",
        "calibration_baseline": "Baseline P95",
        "calibration_target": "Action P10",
        "calibration_samples": "Baseline/action samples",
        "calibration_warning": "No reliable recommendation yet: {warning}",
        "calibration_save": "Save usable calibration results",
        "calibration_saved": (
            "Saved settings for {cues}. Stop and restart the camera to apply them."
        ),
        "calibration_no_results": (
            "Nothing is ready to save. Collect missing phases or improve capture conditions."
        ),
        "calibration_saved_cues": "frown, eye closure, head down",
        "camera_starting": "Starting",
        "camera_running": "Analyzing live video",
        "camera_stopping": "Stopping",
        "camera_stopped": "Stopped",
        "camera_error": "Startup failed",
        "camera_stats": "{frames} frames processed · {fps:.1f} FPS · input {resolution}",
        "camera_resolution_unknown": "resolution pending",
        "top_emotion": "Top expression (model)",
        "confidence": "Top expression probability",
        "output_warning": "Results are written to the selected JSONL file. Raw video is not saved.",
    },
}

STATE_LABELS = {
    "zh": {
        "engaged": "积极参与/感兴趣候选",
        "attentive": "认真听讲候选",
        "confused": "困惑/未听懂候选（需核实）",
        "bored_drowsy_risk": "走神/困倦/无聊候选（需核实）",
        "confusion_cue": "困惑候选线索（需核实）",
        "head_down_cue": "头部角度偏低线索（需核实）",
        "phone_use_cue": "疑似低头看手机/屏幕候选（需核实）",
        "unknown": "证据不足",
    },
    "en": {
        "engaged": "Possible engagement cue (verify)",
        "attentive": "Possible attentive cue (verify)",
        "confused": "Possible confusion cue (verify)",
        "bored_drowsy_risk": "Possible bored/drowsy cue (verify)",
        "confusion_cue": "Possible confusion cue (verify)",
        "head_down_cue": "Low head-angle cue (verify)",
        "phone_use_cue": "Possible phone-use cue (verify)",
        "unknown": "Insufficient evidence",
    },
}

EDUCATIONAL_STATE_LABELS = {
    "zh": {
        "证据不足": "证据不足",
        "积极参与/感兴趣候选（表情线索）": "积极参与/感兴趣候选",
        "认真听讲候选（中性表情线索）": "认真听讲候选",
        "低头看屏幕/疑似使用手机候选（需核实）": "疑似低头看手机/屏幕候选（需核实）",
        "持续低头姿态候选（原因未知，需课堂核实）": "持续低头姿态候选（需核实）",
        "低头伴随持续闭眼候选（困倦或休息等原因需核实）": "低头伴随持续闭眼（困倦候选，需核实）",
        "持续闭眼候选（困倦或休息等原因需核实）": "持续闭眼（困倦候选，需核实）",
        "走神/困倦/无聊候选线索（表情/姿态需核实）": "走神/困倦/无聊候选（需核实）",
        "困惑/未听懂候选线索（需课堂核实）": "困惑/未听懂候选（需核实）",
    },
    "en": {
        "证据不足": "Insufficient evidence",
        "积极参与/感兴趣候选（表情线索）": "Possible engagement cue (verify)",
        "认真听讲候选（中性表情线索）": "Possible attentive cue (verify)",
        "低头看屏幕/疑似使用手机候选（需核实）": "Possible phone-use cue (verify)",
        "持续低头姿态候选（原因未知，需课堂核实）": "Sustained head-down posture (verify cause)",
        "低头伴随持续闭眼候选（困倦或休息等原因需核实）": (
            "Head-down with sustained eye closure (verify)"
        ),
        "持续闭眼候选（困倦或休息等原因需核实）": "Sustained eye closure (verify cause)",
        "走神/困倦/无聊候选线索（表情/姿态需核实）": "Possible bored/drowsy cue (verify)",
        "困惑/未听懂候选线索（需课堂核实）": "Possible confusion cue (verify)",
    },
}

EMOTION_LABELS = {
    "zh": {
        "angry": "愤怒",
        "contempt": "轻蔑",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "开心",
        "neutral": "中性",
        "sad": "悲伤",
        "surprise": "惊讶",
    },
    "en": {
        "angry": "Angry",
        "contempt": "Contempt",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happy",
        "neutral": "Neutral",
        "sad": "Sad",
        "surprise": "Surprise",
    },
}

st.set_page_config(
    page_title="Classroom State Monitor",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172b3a;
        --muted: #53697a;
        --paper: #f2f6fa;
        --line: #d4e0ea;
        --accent: #0071c5;
        --accent-strong: #005a9c;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #e8f0f7; border-right: 1px solid var(--line); }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .intel-logo { display: block; width: 120px; height: 45px; margin-bottom: 18px; }
    .eyebrow {
        color: var(--accent);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }
    .page-title { color: var(--ink); font-size: 34px; font-weight: 700; margin: 2px 0 0; }
    .page-subtitle { color: var(--muted); font-size: 14px; margin: 3px 0 18px; }
    .section-title { color: var(--ink); font-size: 18px; font-weight: 650; margin: 12px 0 4px; }
    .status-live { color: #0071c5; font-weight: 700; }
    .status-idle { color: #8a5a13; font-weight: 700; }
    [data-testid="stProgressBar"] { height: 10px; }
    button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
    button[kind="primary"]:hover {
        background: var(--accent-strong);
        border-color: var(--accent-strong);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    browser_client_id = st.session_state.setdefault(
        "browser_camera_client_id", uuid.uuid4().hex[:12]
    )
    if "browser_jsonl_path" not in st.session_state:
        st.session_state.browser_jsonl_path = (
            f"outputs/live_predictions_{browser_client_id}.jsonl"
        )
    previous_processor = st.session_state.get("browser_camera_processor")
    if previous_processor is not None and previous_processor.is_closed:
        st.session_state.browser_camera_processor = None
        st.session_state.browser_jsonl_path = (
            f"outputs/live_predictions_{browser_client_id}_{uuid.uuid4().hex[:8]}.jsonl"
        )
    st.markdown(
        '<svg class="intel-logo" viewBox="0 0 120 45" role="img" aria-label="Intel">'
        '<rect width="120" height="45" fill="#0071c5"/>'
        '<ellipse cx="60" cy="23" rx="52" ry="19" fill="none" stroke="#fff" '
        'stroke-width="1.5" transform="rotate(-9 60 23)"/>'
        '<text x="19" y="32" fill="#fff" font-family="Arial, sans-serif" '
        'font-size="29" font-weight="600">intel</text></svg>',
        unsafe_allow_html=True,
    )
    language_choice = st.selectbox(
        "语言 / Language",
        options=["中文", "English"],
        disabled=st.session_state.get("browser_camera_active", False),
    )
    language_code = "zh" if language_choice == "中文" else "en"
    text = TEXT[language_code]
    st.markdown(
        f'<div class="eyebrow">{text["sidebar_eyebrow"]}</div>',
        unsafe_allow_html=True,
    )
    st.subheader(text["sidebar_title"])
    jsonl_path = st.text_input(
        text["path"],
        key="browser_jsonl_path",
        help=text["path_help"],
        disabled=st.session_state.get("browser_camera_active", False),
    )
    refresh_seconds = st.selectbox(
        text["refresh"],
        options=[1, 2, 5],
        index=0,
        format_func=lambda value: text["refresh_option"].format(seconds=value),
    )
    st.caption(text["privacy"])

st.markdown(f'<div class="eyebrow">{text["eyebrow"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-title">{text["title"]}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-subtitle">{text["subtitle"]}</div>',
    unsafe_allow_html=True,
)


def _update_detector_path() -> None:
    st.session_state.live_detector_model = (
        "models/weights/scrfd_2.5g.onnx"
        if st.session_state.live_detector_type == "scrfd_2_5g"
        else "models/weights/yunet.onnx"
    )


@st.fragment(run_every=f"{refresh_seconds}s")
def render_monitor(path_text: str, locale: str) -> None:
    text = TEXT[locale]
    browser_client_id = st.session_state.browser_camera_client_id
    camera_active = st.session_state.get("browser_camera_active", False)

    with st.expander(text["camera_title"], expanded=True):
        setting_columns = st.columns([1, 0.8, 1.4])
        detector_type = setting_columns[0].selectbox(
            text["detector_type"],
            options=["yunet", "scrfd_2_5g"],
            index=0,
            format_func=lambda value: "YuNet" if value == "yunet" else "SCRFD-2.5G",
            disabled=camera_active,
            key="live_detector_type",
            on_change=_update_detector_path,
        )
        device = setting_columns[1].selectbox(
            text["device"],
            options=["GPU", "CPU"],
            index=0,
            disabled=camera_active,
            key="live_camera_device",
        )
        model_columns = setting_columns[2].columns(2)
        detector_model = model_columns[0].text_input(
            text["detector_model"],
            value="models/weights/yunet.onnx",
            disabled=camera_active,
            key="live_detector_model",
        )
        emotion_model = model_columns[1].text_input(
            text["emotion_model"],
            value=DEFAULT_EMOTION_MODEL,
            disabled=camera_active,
            key="live_emotion_model",
        )
        identity_model = st.text_input(
            text["identity_model"],
            value="models/weights/face-reidentification-retail-0095.xml",
            disabled=camera_active,
            key="live_identity_model",
        )
        json_endpoint = st.text_input(
            text["json_endpoint"],
            value="",
            help=text["json_endpoint_help"],
            disabled=camera_active,
            key="live_json_endpoint",
        )
        try:
            from streamlit_webrtc import webrtc_streamer
        except ImportError:
            st.error("Install browser camera support with `pip install -e '.[dashboard]'`.")
            return

        from classroom_emotion.live_camera import BrowserCameraProcessor

        endpoint = json_endpoint.strip() or None
        processor_signature = (
            browser_client_id,
            detector_model,
            emotion_model,
            identity_model,
            device,
            path_text,
            locale,
            detector_type,
            endpoint,
        )
        processor = st.session_state.get("browser_camera_processor")
        if (
            processor is None
            or processor.is_closed
            or processor.signature != processor_signature
        ):
            if processor is not None:
                processor.close()
            processor = BrowserCameraProcessor(
                client_id=browser_client_id,
                detector_model=detector_model,
                emotion_model=emotion_model,
                identity_model=identity_model,
                device=device,
                output_path=path_text,
                language=locale,
                detector_type=detector_type,
                json_endpoint=endpoint,
            )
            st.session_state.browser_camera_processor = processor

        rtc_context = webrtc_streamer(
            key=f"browser-camera-{browser_client_id}",
            video_frame_callback=processor.receive,
            on_video_ended=processor.close,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": CAMERA_INPUT.width},
                    "height": {"ideal": CAMERA_INPUT.height},
                    "frameRate": {"ideal": CAMERA_INPUT.fps, "max": CAMERA_INPUT.fps},
                },
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )
        camera_active = bool(rtc_context.state.playing)
        st.session_state.browser_camera_active = camera_active
        st.caption(text["output_warning"])
        camera_status = processor.snapshot()
        if not camera_status["identity_reidentification"]:
            st.warning(text["identity_warning"])
        st.caption(text["identity_scope"])
        if camera_status["frame_count"]:
            path_text = camera_status["output_path"]
        if camera_status["error"]:
            st.error(camera_status["error"])
        elif camera_active:
            capture_width = camera_status["capture_width"]
            capture_height = camera_status["capture_height"]
            resolution = (
                f"{capture_width}x{capture_height}"
                if capture_width and capture_height
                else text["camera_resolution_unknown"]
            )
            st.success(
                text["camera_stats"].format(
                    frames=camera_status["frame_count"],
                    fps=camera_status["fps"],
                    resolution=resolution,
                )
            )
        else:
            st.info(text["camera_idle"])
        if camera_status.get("delivery") is not None:
            st.caption(text["delivery_stats"].format(**camera_status["delivery"]))
            if camera_status["delivery"]["last_error"]:
                st.warning(
                    text["delivery_error"].format(
                        error=camera_status["delivery"]["last_error"]
                    )
                )
        _render_cue_calibration(processor, camera_active, locale)

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    file_stat = None
    records: list[dict[str, Any]] = []
    skipped_lines = 0
    try:
        file_stat = path.stat()
    except FileNotFoundError:
        pass
    except OSError as error:
        st.error(text["read_error"].format(error=error))
        return

    if file_stat is not None:
        try:
            records, skipped_lines = _read_records(path)
        except OSError as error:
            st.error(text["read_error"].format(error=error))
            return
        last_write = datetime.fromtimestamp(file_stat.st_mtime).astimezone()
        seconds_since_write = (datetime.now().astimezone() - last_write).total_seconds()
        recently_updated = seconds_since_write <= max(5, refresh_seconds * 3)
        status_class = "status-live" if recently_updated else "status-idle"
        status_text = text["recent_write"] if recently_updated else text["no_recent_write"]
        status_column, file_column = st.columns([1, 2])
        with status_column:
            st.markdown(
                f'<span class="{status_class}">● {status_text}</span>',
                unsafe_allow_html=True,
            )
        with file_column:
            st.caption(
                f'{text["source"]}: {escape(path.name)} · '
                f'{text["updated"]} {last_write:%H:%M:%S}'
            )
    else:
        st.info(text["missing_file"].format(path=path))

    if not records:
        st.info(text["empty_file"])
        return

    latest_by_track: dict[int, dict[str, Any]] = {}
    for record in records:
        track_id = int(record["track_id"])
        previous = latest_by_track.get(track_id)
        if previous is None or _record_order(record) >= _record_order(previous):
            latest_by_track[track_id] = record

    latest_records = sorted(
        latest_by_track.values(), key=lambda record: int(record["track_id"])
    )
    latest_time = max(float(record["timestamp_seconds"]) for record in records)
    latest_frame = max(int(record["frame_index"]) for record in records)
    state_counts: dict[str, int] = {}
    for record in latest_records:
        label = _educational_state_display(record, locale)
        state_counts[label] = state_counts.get(label, 0) + 1

    metrics = st.columns(4)
    metrics[0].metric(text["tracks"], len(latest_records))
    metrics[1].metric(text["observations"], len(records))
    metrics[2].metric(text["video_time"], f'{latest_time:.1f} {text["seconds"]}')
    metrics[3].metric(text["latest_frame"], latest_frame)

    if skipped_lines:
        st.warning(text["skipped"].format(count=skipped_lines))

    st.markdown(
        f'<div class="section-title">{text["state_distribution"]}</div>',
        unsafe_allow_html=True,
    )
    max_count = max(state_counts.values(), default=0)
    for state, count in state_counts.items():
        if count == 0:
            continue
        ratio = count / max_count if max_count else 0.0
        st.progress(ratio, text=f'{state} · {count}')

    st.markdown(
        f'<div class="section-title">{text["track_states"]}</div>',
        unsafe_allow_html=True,
    )
    track_filter = st.segmented_control(
        text["track_filter"],
        options=[
            text["track_filter_all"],
            text["track_filter_candidate"],
            text["track_filter_unknown"],
        ],
        default=text["track_filter_all"],
        key=f"track_filter_{locale}",
    )
    filtered_records = [
        record
        for record in latest_records
        if track_filter == text["track_filter_all"]
        or (record.get("state") == "unknown")
        == (track_filter == text["track_filter_unknown"])
    ]
    selected_track = None
    if filtered_records:
        track_rows = [
            {
                text["track_id"]: int(record["track_id"]),
                text["state"]: _educational_state_display(record, locale),
                text["visual_evidence"]: record.get("visual_evidence", text["unknown"]),
                text["observation_count"]: int(record.get("observation_count", 0)),
                text["timestamp"]: round(float(record["timestamp_seconds"]), 2),
            }
            for record in filtered_records
        ]
        table_state = st.dataframe(
            track_rows,
            hide_index=True,
            width="stretch",
            alt=text["track_states"],
            on_select="rerun",
            selection_mode="single-row-required",
            selection_default={"selection": {"rows": [0]}},
            key=f"track_table_{locale}_{track_filter}",
        )
        selected_row = table_state.selection.rows[0]
        selected_track = int(filtered_records[selected_row]["track_id"])
        _render_track_detail(records, selected_track, locale)
    else:
        st.info(text["no_tracks_for_filter"])

    st.markdown(
        f'<div class="section-title">{text["recent_observations"]}</div>',
        unsafe_allow_html=True,
    )
    recent_records = sorted(records, key=_record_order, reverse=True)[:20]
    st.dataframe(
        [
            {
                text["timestamp"]: round(float(record["timestamp_seconds"]), 2),
                text["frame"]: int(record["frame_index"]),
                text["track_id"]: int(record["track_id"]),
                text["state"]: _educational_state_display(record, locale),
                text["observation_count"]: int(record.get("observation_count", 0)),
            }
            for record in recent_records
        ],
        hide_index=True,
        width="stretch",
        alt=text["recent_observations"],
    )
    st.caption(text["disclaimer"])


def _render_track_detail(records: list[dict[str, Any]], selected_track: int, locale: str) -> None:
    text = TEXT[locale]
    selected_records = [record for record in records if int(record["track_id"]) == selected_track]
    latest_record = max(selected_records, key=_record_order)
    emotion, confidence = _emotion_display(latest_record, locale)
    emotion_columns = st.columns(2)
    emotion_columns[0].metric(text["top_emotion"], emotion)
    emotion_columns[1].metric(text["confidence"], f"{confidence:.0%}")

    st.markdown(
        f'<div class="section-title">{text["facial_cues_title"]}</div>',
        unsafe_allow_html=True,
    )
    cue_columns = st.columns(2)
    cue_columns[0].metric(
        text["eye_closed_cue"],
        _facial_cue_display(latest_record, "eye_closed_score", "eyes_closed", locale),
    )
    cue_columns[1].metric(
        text["frown_cue"],
        _facial_cue_display(latest_record, "frown_score", "frown", locale),
    )
    st.caption(
        f'{text["head_pose_angles"]}: '
        f'{_head_pose_display(latest_record, locale)}'
    )
    st.markdown(
        f'<div class="section-title">{text["assessment_title"]}</div>',
        unsafe_allow_html=True,
    )
    st.write(
        f'**{text["state"]}:** '
        f'{_educational_state_display(latest_record, locale)}'
    )
    secondary_state = latest_record.get("secondary_state")
    st.write(
        f'**{text["secondary_state"]}:** '
        f'{secondary_state if secondary_state else text["no_secondary_state"]}'
    )
    st.write(
        f'**{text["visual_evidence"]}:** '
        f'{latest_record.get("visual_evidence", text["unknown"])}'
    )
    st.write(
        f'**{text["cognitive_load"]}:** '
        f'{latest_record.get("cognitive_load", text["unavailable_load"])}'
    )
    st.write(
        f'**{text["teacher_advice"]}:** '
        f'{latest_record.get("teacher_advice", text["unknown"])}'
    )
    emotion_labels = sorted(
        {
            label
            for record in selected_records
            for label in record.get("probabilities", {})
        }
    )
    st.markdown(
        f'<div class="section-title">{text["emotion_trend"]}</div>',
        unsafe_allow_html=True,
    )
    if emotion_labels:
        chart_rows = {
            label: [
                float(record.get("probabilities", {}).get(label, 0.0))
                for record in selected_records
            ]
            for label in emotion_labels
        }
        chart_rows = {
            EMOTION_LABELS[locale].get(label, label): values
            for label, values in chart_rows.items()
        }
        st.line_chart(chart_rows, height=260)


def _emotion_display(record: dict[str, Any], locale: str) -> tuple[str, float]:
    probabilities = record.get("probabilities", {})
    if not isinstance(probabilities, dict) or not probabilities:
        return TEXT[locale]["unknown"], 0.0
    label, confidence = max(probabilities.items(), key=lambda item: item[1])
    return EMOTION_LABELS[locale].get(str(label), str(label)), float(confidence)


def _facial_cue_display(
    record: dict[str, Any], cue_name: str, evidence_name: str, locale: str
) -> str:
    cues = record.get("facial_cues", {})
    score = cues.get(cue_name) if isinstance(cues, dict) else None
    if score is None:
        return TEXT[locale]["cue_unavailable"]
    status = (
        TEXT[locale]["cue_triggered"]
        if evidence_name in record.get("evidence", [])
        else TEXT[locale]["cue_below_threshold"]
    )
    return f"{float(score):.2f} · {status}"


def _head_pose_display(record: dict[str, Any], locale: str) -> str:
    cues = record.get("facial_cues", {})
    if not isinstance(cues, dict):
        return TEXT[locale]["cue_unavailable"]
    values = tuple(
        cues.get(name)
        for name in (
            "head_pitch_degrees",
            "head_yaw_degrees",
            "head_roll_degrees",
        )
    )
    if any(value is None for value in values):
        return TEXT[locale]["cue_unavailable"]
    pitch, yaw, roll = (float(value) for value in values)
    return f"{pitch:+.1f}° · {yaw:+.1f}° · {roll:+.1f}°"


def _educational_state_display(record: dict[str, Any], locale: str) -> str:
    primary_state = record.get("primary_state")
    if isinstance(primary_state, str) and primary_state:
        display = EDUCATIONAL_STATE_LABELS[locale].get(primary_state, primary_state)
    else:
        state = str(record.get("state", "unknown"))
        display = STATE_LABELS[locale].get(state, TEXT[locale]["unknown"])
    if "state_held" in record.get("evidence", []):
        return f"{display}（保持中）" if locale == "zh" else f"{display} (held)"
    return display


def _record_order(record: dict[str, Any]) -> tuple[float, int]:
    return float(record.get("timestamp_seconds", 0.0)), int(record.get("frame_index", 0))


def _read_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    text = path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    skipped_lines = 0
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            for field in ("frame_index", "track_id", "timestamp_seconds", "state"):
                if field not in record:
                    raise ValueError(f"missing {field}")
            records.append(record)
        except (json.JSONDecodeError, TypeError, ValueError):
            is_incomplete_tail = line_number == len(lines) and not text.endswith("\n")
            if not is_incomplete_tail:
                skipped_lines += 1
    return records, skipped_lines


def _render_cue_calibration(processor, camera_active: bool, locale: str) -> None:
    from classroom_emotion.calibration import recommend_cue_calibration, save_cue_calibration

    text = TEXT[locale]
    snapshot = processor.calibration_snapshot()
    track_ids = snapshot["track_ids"]
    with st.container(border=True):
        st.subheader(text["calibration_title"])
        st.caption(text["calibration_instructions"])
        if not track_ids:
            processor.configure_calibration(None, "off")
            st.info(text["calibration_waiting_track"])
            return

        controls = st.columns([1, 1.2])
        track_id = controls[0].selectbox(
            text["calibration_track"],
            options=track_ids,
            key=f"calibration_track_{processor.client_id}",
            disabled=not camera_active,
        )
        phases = ("off", "baseline", "frown", "eyes_closed", "head_down")
        phase = controls[1].selectbox(
            text["calibration_phase"],
            options=phases,
            format_func=lambda value: text[f"calibration_phase_{value}"],
            key=f"calibration_phase_{processor.client_id}",
            disabled=not camera_active,
        )
        processor.configure_calibration(int(track_id), phase)
        snapshot = processor.calibration_snapshot()
        samples = [
            sample for sample in snapshot["samples"] if sample["track_id"] == int(track_id)
        ]
        st.caption(text["calibration_counts"].format(count=len(samples)))
        recommendations = recommend_cue_calibration(samples)

        cue_labels = {
            "frown_score": text["calibration_phase_frown"],
            "eye_closed_score": text["calibration_phase_eyes_closed"],
            "head_down_score": text["calibration_phase_head_down"],
        }
        table_rows = []
        for cue, result in recommendations.items():
            table_rows.append(
                {
                    text["calibration_phase"]: cue_labels[cue],
                    text["calibration_samples"]: (
                        f"{result.baseline_samples}/{result.target_samples}"
                    ),
                    text["calibration_baseline"]: (
                        f"{result.baseline_p95:.3f}" if result.baseline_p95 is not None else "-"
                    ),
                    text["calibration_target"]: (
                        f"{result.target_p10:.3f}" if result.target_p10 is not None else "-"
                    ),
                    text["calibration_threshold"]: (
                        f"{result.threshold:.3f}" if result.threshold is not None else "-"
                    ),
                    text["calibration_duration"]: (
                        f"{result.persistence_seconds:.2f}"
                        if result.persistence_seconds is not None
                        else "-"
                    ),
                }
            )
        st.dataframe(
            table_rows,
            hide_index=True,
            width="stretch",
            alt=text["calibration_title"],
        )
        for cue, result in recommendations.items():
            if result.warning:
                st.caption(
                    f'{cue_labels[cue]}: '
                    + text["calibration_warning"].format(warning=result.warning)
                )

        has_recommendation = any(
            result.threshold is not None and result.persistence_seconds is not None
            for result in recommendations.values()
        )
        if st.button(
            text["calibration_save"],
            key=f"calibration_save_{processor.client_id}",
            disabled=not has_recommendation,
        ):
            applied = save_cue_calibration(
                "configs/cue_calibration.yaml", recommendations
            )
            if applied:
                st.success(
                    text["calibration_saved"].format(
                        cues=text["calibration_saved_cues"]
                    )
                )
            else:
                st.warning(text["calibration_no_results"])


render_monitor(jsonl_path, language_code)

with st.container(border=True):
    st.subheader(text["mapping_title"])
    st.markdown("\n".join(f"- {item}" for item in text["mapping_rows"]))
    st.caption(text["mapping_temporal"])