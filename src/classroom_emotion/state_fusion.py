"""Conservative mapping from observable face signals to classroom states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from classroom_emotion.domain import ClassroomState, FaceCues, StateEstimate

EMOTION_TO_CLASSROOM_STATE = {
    "happy": ClassroomState.ENGAGED,
    "surprise": ClassroomState.ENGAGED,
    "neutral": ClassroomState.ATTENTIVE,
    "confused": ClassroomState.CONFUSED,
    "sad": ClassroomState.BORED_DROWSY_RISK,
    "disgust": ClassroomState.BORED_DROWSY_RISK,
}

EMOTION_LABELS_ZH = {
    "happy": "高兴",
    "surprise": "惊讶",
    "neutral": "中性",
    "confused": "困惑",
    "sad": "悲伤",
    "disgust": "厌恶",
}


@dataclass(frozen=True, slots=True)
class FusionThresholds:
    engaged: float = 0.55
    attentive: float = 0.55
    confused: float = 0.55
    bored_drowsy: float = 0.55
    frown_cue: float = 0.13
    eye_closed_cue: float = 0.50
    head_down_cue: float = 0.13
    head_down_pitch_degrees: float = 20.0


class ClassroomStateFusion:
    def __init__(self, thresholds: FusionThresholds | None = None) -> None:
        self._thresholds = thresholds or FusionThresholds()

    def classify(
        self,
        *,
        track_id: int,
        timestamp_seconds: float,
        probabilities: Mapping[str, float],
        observation_count: int,
        cues: FaceCues | None = None,
        sustained_cues: frozenset[str] | None = None,
        window_complete: bool = False,
    ) -> StateEstimate:
        cues = cues or FaceCues()
        evidence: list[str] = []

        if sustained_cues is None:
            head_down_active = (
                cues.head_pitch_degrees <= -self._thresholds.head_down_pitch_degrees
                if cues.head_pitch_degrees is not None
                else cues.head_down_score is not None
                and cues.head_down_score >= self._thresholds.head_down_cue
            )
            eyes_closed_active = (
                cues.eye_closed_score is not None
                and cues.eye_closed_score >= self._thresholds.eye_closed_cue
            )
        else:
            head_down_active = "head_down" in sustained_cues
            eyes_closed_active = "eyes_closed" in sustained_cues

        if eyes_closed_active:
            evidence.append("eyes_closed")
            if head_down_active:
                evidence.append("head_down")
            return self._estimate(
                track_id,
                timestamp_seconds,
                ClassroomState.BORED_DROWSY_RISK,
                probabilities,
                observation_count,
                evidence,
                cues=cues,
                primary_state=(
                    "低头伴随持续闭眼候选（困倦或休息等原因需核实）"
                    if head_down_active
                    else "持续闭眼候选（困倦或休息等原因需核实）"
                ),
                visual_evidence=(
                    "检测到持续闭眼线索；闭眼可能来自眨眼、休息或其他原因，"
                    "不能单独确认困倦。"
                ),
                teacher_advice=(
                    "先通过课堂任务或轻声确认学生是否需要帮助、调整坐姿或短暂休息，"
                    "不要仅凭姿态给学生下结论。"
                ),
            )

        if head_down_active:
            evidence.append("head_down")
            pose_text = (
                f"窗口平均 Pitch {cues.head_pitch_degrees:.1f}°"
                if cues.head_pitch_degrees is not None
                else "持续头部俯角偏低"
            )
            neutral_score = probabilities.get("neutral", 0.0)
            neutral_is_dominant = (
                neutral_score >= self._thresholds.attentive
                and all(
                    neutral_score >= probability
                    for label, probability in probabilities.items()
                    if label != "neutral"
                )
            )
            eyes_open_active = (
                cues.eye_closed_score is not None
                and cues.eye_closed_score < self._thresholds.eye_closed_cue
            )
            if (
                sustained_cues is not None
                and eyes_open_active
                and neutral_is_dominant
            ):
                evidence.extend(("eyes_open", "emotion:neutral"))
                return self._estimate(
                    track_id,
                    timestamp_seconds,
                    ClassroomState.PHONE_USE_CUE,
                    probabilities,
                    observation_count,
                    evidence,
                    cues=cues,
                    primary_state="低头看屏幕/疑似使用手机候选（需核实）",
                    visual_evidence=(
                        f"{pose_text}，眼睛睁开且中性表情占优；当前未检测手机或手部动作，"
                        "也可能是在看书或书写，不能确认手机使用。"
                    ),
                    teacher_advice=(
                        "结合课堂任务和手部、桌面情况核实；若需确认手机使用，"
                        "应增加手机或手部目标检测。"
                    ),
                )
            return self._estimate(
                track_id,
                timestamp_seconds,
                ClassroomState.HEAD_DOWN_CUE,
                probabilities,
                observation_count,
                evidence,
                cues=cues,
                primary_state="持续低头姿态候选（原因未知，需课堂核实）",
                visual_evidence=(
                    f"{pose_text}；姿态只能说明头部方向，不能证明在看手机或打瞌睡。"
                ),
                teacher_advice=(
                    "结合课堂任务观察学生是否在看材料、书写或需要帮助，"
                    "不要仅凭头部姿态推断手机使用。"
                ),
            )

        frown_active = (
            cues.frown_score is not None
            and cues.frown_score >= self._thresholds.frown_cue
            if sustained_cues is None
            else "frown" in sustained_cues
        )
        if frown_active:
            evidence.append("frown")
            return self._estimate(
                track_id,
                timestamp_seconds,
                ClassroomState.CONFUSED,
                probabilities,
                observation_count,
                evidence,
                cues=cues,
                primary_state="困惑/未听懂候选线索（需课堂核实）",
                visual_evidence=(
                    "检测到持续的眉部下压线索；仍需结合学生口头说明或作答过程核实。"
                ),
                teacher_advice=(
                    "用一个不带评价的追问请学生说明当前思路，再按其回答提供提示或例题。"
                ),
            )

        label, score = max(
            (
                (label, probability)
                for label, probability in probabilities.items()
                if label in EMOTION_TO_CLASSROOM_STATE
            ),
            key=lambda item: item[1],
            default=("", 0.0),
        )
        state = EMOTION_TO_CLASSROOM_STATE.get(label)
        threshold = {
            ClassroomState.ENGAGED: self._thresholds.engaged,
            ClassroomState.ATTENTIVE: self._thresholds.attentive,
            ClassroomState.CONFUSED: self._thresholds.confused,
            ClassroomState.BORED_DROWSY_RISK: self._thresholds.bored_drowsy,
        }.get(state, 1.0)
        if state is not None and (score >= threshold or window_complete):
            evidence.append(f"emotion:{label}")
            low_confidence = score < threshold
            if low_confidence:
                evidence.append("low_confidence")
            emotion_name = EMOTION_LABELS_ZH[label]
            if state is ClassroomState.CONFUSED:
                primary_state = "困惑/未听懂候选线索（需课堂核实）"
                visual_evidence = (
                    f"表情模型当前最高类别为{emotion_name}；"
                    "仍需结合学生口头说明或作答过程核实。"
                )
                teacher_advice = (
                    "用一个不带评价的追问请学生说明当前思路，"
                    "再按其回答提供提示或例题。"
                )
            elif state is ClassroomState.BORED_DROWSY_RISK:
                primary_state = "走神/困倦/无聊候选线索（表情/姿态需核实）"
                visual_evidence = (
                    f"表情模型当前最高类别为{emotion_name}；"
                    "这不能单独证明走神、困倦或无聊。"
                )
                teacher_advice = (
                    "结合学生作答、视线和课堂互动核实状态，"
                    "再提供简短提示或休息选择。"
                )
            elif state is ClassroomState.ENGAGED:
                primary_state = "积极参与/感兴趣候选（表情线索）"
                visual_evidence = (
                    f"表情模型当前最高类别为{emotion_name}；"
                    "这不能单独证明学生正在投入学习。"
                )
                teacher_advice = (
                    "可顺势提供拓展问题，并通过学生的回答确认其兴趣和理解情况。"
                )
            else:
                primary_state = "认真听讲候选（中性表情线索）"
                visual_evidence = (
                    "表情模型当前最高类别为中性；当前系统未测量视线，"
                    "无法仅凭中性表情确认认真听讲。"
                )
                teacher_advice = (
                    "结合课堂提问或随堂作答确认理解情况，"
                    "不要把中性表情直接视为专注。"
                )
            if low_confidence:
                visual_evidence = (
                    f"三秒窗口内最高可映射类别为{emotion_name}，平均概率为{score:.0%}，"
                    f"低于参考阈值{threshold:.0%}；仅作为低置信候选，需结合课堂互动核实。"
                )
            return self._estimate(
                track_id,
                timestamp_seconds,
                state,
                probabilities,
                observation_count,
                evidence,
                cues=cues,
                primary_state=primary_state,
                visual_evidence=visual_evidence,
                teacher_advice=teacher_advice,
            )

        return self._estimate(
            track_id,
            timestamp_seconds,
            ClassroomState.UNKNOWN,
            probabilities,
            observation_count,
            evidence,
            cues=cues,
        )

    @staticmethod
    def _estimate(
        track_id: int,
        timestamp_seconds: float,
        state: ClassroomState,
        probabilities: Mapping[str, float],
        observation_count: int,
        evidence: list[str],
        *,
        cues: FaceCues,
        primary_state: str = "证据不足",
        visual_evidence: str = "当前系统未充分测量视线、眼睑和身体姿态，无法可靠判断教学状态。",
        teacher_advice: str = "通过开放式提问或作答过程核实是否需要支持，不要仅凭面部表情判断。",
    ) -> StateEstimate:
        return StateEstimate(
            track_id=track_id,
            timestamp_seconds=timestamp_seconds,
            state=state,
            probabilities=probabilities,
            observation_count=observation_count,
            cues=cues,
            evidence=tuple(evidence),
            primary_state=primary_state,
            secondary_state=None,
            visual_evidence=visual_evidence,
            cognitive_load="无法判断",
            teacher_advice=teacher_advice,
        )
