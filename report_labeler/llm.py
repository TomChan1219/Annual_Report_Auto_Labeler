from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from report_labeler.models import JudgmentRecord, ModelConfig, SentenceRecord


SYSTEM_PROMPT = """你是一名严谨的中文企业年报标注助手。你的任务是判断一个句子是否说明企业在该年度已经将工业互联网平台相关能力真正用在了自身业务流程中。

请严格依据以下标准判断：
1. 判为1：句子描述企业自身在生产、制造、研发、供应链、运维、质量控制等内部环节，已经采用、部署、实施、应用、建成、上线、实现了相关能力。
2. 判为0：句子如果只是未来规划、战略布局、探索推进、技术研发、平台建设、对外销售方案、赋能客户、通用办公系统，或没有明确落地行为，都判为0。
3. 优先关注“自身应用”而非“对外服务”。
4. 如果句子有具体动作词和可验证效果，如“实现了…提升了…降低了…监测了…”，更接近1。

输出必须是 JSON 对象，只包含以下字段：
- label: 0 或 1
- confidence: 0 到 1 之间的小数
- reason: 一句中文理由，风格尽量像人工批注，例如“这句话判定为1，原因是……”
"""


@dataclass
class ModelResult:
    label: int | None
    confidence: float
    reason: str
    source: str


class BaseJudge:
    def judge(self, sentence_record: SentenceRecord, model_config: ModelConfig) -> ModelResult:
        raise NotImplementedError


class MockJudge(BaseJudge):
    def judge(self, sentence_record: SentenceRecord, model_config: ModelConfig) -> ModelResult:
        if sentence_record.rule_label == 1:
            reason = (
                f"这句话判定为1，原因是它明确描述了企业自身已经采取了落地动作，"
                f"并且出现了可验证的工业场景或效果信号：{sentence_record.rule_reason}"
            )
            return ModelResult(1, max(sentence_record.rule_confidence, 0.82), reason, "mock")

        if sentence_record.rule_label == 0:
            reason = (
                f"这句话判定为0，原因是它更像规划、平台建设、对外服务或缺少明确落地动作："
                f"{sentence_record.rule_reason}"
            )
            return ModelResult(0, max(sentence_record.rule_confidence, 0.8), reason, "mock")

        label = 1 if "self_use_action" in sentence_record.rule_flags and "industrial_scene" in sentence_record.rule_flags else 0
        reason = (
            "这句话判定为1，原因是它包含自身应用动作和工业场景。"
            if label == 1
            else "这句话判定为0，原因是规则未找到足够的自身落地证据。"
        )
        return ModelResult(label, 0.55, reason, "mock")


class DeepSeekJudge(BaseJudge):
    def judge(self, sentence_record: SentenceRecord, model_config: ModelConfig) -> ModelResult:
        if not model_config.api_key:
            return ModelResult(None, 0.0, "未提供 DeepSeek API Key，无法调用模型。", "deepseek")

        payload = {
            "model": model_config.model_name,
            "temperature": model_config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(sentence_record)},
            ],
        }
        headers = {
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            model_config.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=model_config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_model_json(content)
        return ModelResult(
            label=normalize_label(parsed.get("label")),
            confidence=float(parsed.get("confidence", 0.5)),
            reason=str(parsed.get("reason", "")).strip() or "模型未返回判断理由。",
            source="deepseek",
        )


def get_judge(model_config: ModelConfig) -> BaseJudge:
    if model_config.provider == "deepseek":
        return DeepSeekJudge()
    return MockJudge()


def build_user_prompt(sentence_record: SentenceRecord) -> str:
    payload = {
        "sentence": sentence_record.sentence,
        "context_before": sentence_record.context_before,
        "context_after": sentence_record.context_after,
        "matched_keywords": sentence_record.matched_keywords,
        "keyword_categories": sentence_record.keyword_categories,
        "rule_flags": sentence_record.rule_flags,
        "rule_hint_label": sentence_record.rule_label,
        "rule_hint_reason": sentence_record.rule_reason,
        "company_name": sentence_record.company_name,
        "year": sentence_record.year,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_model_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def normalize_label(value: object) -> int | None:
    if value in {0, "0"}:
        return 0
    if value in {1, "1"}:
        return 1
    return None


def merge_judgment(sentence_record: SentenceRecord, model_result: ModelResult) -> JudgmentRecord:
    if sentence_record.rule_label is not None and sentence_record.rule_confidence >= 0.92:
        final_label = sentence_record.rule_label
        confidence = sentence_record.rule_confidence
        reason = (
            f"这句话判定为{final_label}，原因是规则已经给出了高确定性判断："
            f"{sentence_record.rule_reason}"
        )
        source = "rules"
    else:
        fallback_label = sentence_record.rule_label if sentence_record.rule_label is not None else 0
        final_label = model_result.label if model_result.label is not None else fallback_label
        confidence = max(model_result.confidence, sentence_record.rule_confidence)
        reason = model_result.reason or sentence_record.rule_reason
        source = model_result.source

    return JudgmentRecord(
        record_id=sentence_record.record_id,
        source_file=sentence_record.source_file,
        stock_id=sentence_record.stock_id,
        year=sentence_record.year,
        company_name=sentence_record.company_name,
        sentence=sentence_record.sentence,
        sentence_index=sentence_record.sentence_index,
        char_position=sentence_record.char_position,
        context_before=sentence_record.context_before,
        context_after=sentence_record.context_after,
        matched_keywords=sentence_record.matched_keywords,
        keyword_categories=sentence_record.keyword_categories,
        rule_flags=sentence_record.rule_flags,
        rule_label=sentence_record.rule_label,
        model_label=model_result.label,
        final_label=final_label,
        confidence=round(min(max(confidence, 0.0), 1.0), 3),
        judge_reason=reason,
        model_source=source,
    )
