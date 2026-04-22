from __future__ import annotations

import re
from dataclasses import dataclass

from report_labeler.keywords import (
    BACKGROUND_WORDS,
    EFFECT_WORDS,
    FUTURE_WORDS,
    GENERIC_SYSTEM_WORDS,
    INDUSTRIAL_SCENE_WORDS,
    OUTBOUND_WORDS,
    PLATFORM_BUILD_WORDS,
    PRIMARY_KEYWORDS_BY_CATEGORY,
    SECONDARY_KEYWORDS_BY_CATEGORY,
    SELF_USE_WORDS,
)


def normalize_keyword_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，,。！？；;：:、（）()【】\[\]《》“”\"'‘’\-—_/·]", "", text)
    return text.upper()


PRIMARY_KEYWORDS = {
    keyword: category
    for category, words in PRIMARY_KEYWORDS_BY_CATEGORY.items()
    for keyword in words
}

SECONDARY_KEYWORDS = {
    keyword: category
    for category, words in SECONDARY_KEYWORDS_BY_CATEGORY.items()
    for keyword in words
}

NORMALIZED_PRIMARY_KEYWORDS = {
    keyword: normalize_keyword_text(keyword)
    for keyword in PRIMARY_KEYWORDS
}

NORMALIZED_SECONDARY_KEYWORDS = {
    keyword: normalize_keyword_text(keyword)
    for keyword in SECONDARY_KEYWORDS
}


@dataclass
class RuleEvaluation:
    matched_keywords: list[str]
    keyword_categories: list[str]
    flags: list[str]
    label: int | None
    confidence: float
    reason: str


def match_keywords(sentence: str) -> tuple[list[str], list[str], list[str]]:
    normalized_sentence = normalize_keyword_text(sentence)
    primary_matches = [
        keyword for keyword, normalized_keyword in NORMALIZED_PRIMARY_KEYWORDS.items()
        if normalized_keyword and normalized_keyword in normalized_sentence
    ]
    secondary_matches = [
        keyword
        for keyword, normalized_keyword in NORMALIZED_SECONDARY_KEYWORDS.items()
        if normalized_keyword and normalized_keyword in normalized_sentence and keyword not in primary_matches
    ]
    matched = primary_matches + secondary_matches
    categories = sorted(
        {
            *(PRIMARY_KEYWORDS[keyword] for keyword in primary_matches),
            *(SECONDARY_KEYWORDS[keyword] for keyword in secondary_matches),
        }
    )
    return matched, categories, secondary_matches


def evaluate_rules(sentence: str) -> RuleEvaluation:
    matched_keywords, categories, secondary_matches = match_keywords(sentence)
    flags: list[str] = []

    has_future = contains_any(sentence, FUTURE_WORDS)
    has_self_use = contains_any(sentence, SELF_USE_WORDS)
    has_outbound = contains_any(sentence, OUTBOUND_WORDS)
    has_effect = contains_any(sentence, EFFECT_WORDS)
    has_build = contains_any(sentence, PLATFORM_BUILD_WORDS)
    has_generic = contains_any(sentence, GENERIC_SYSTEM_WORDS)
    has_industrial_scene = contains_any(sentence, INDUSTRIAL_SCENE_WORDS)
    has_background = contains_any(sentence, BACKGROUND_WORDS)

    if secondary_matches:
        flags.append("secondary_keyword_hit")
    if matched_keywords and not secondary_matches:
        flags.append("primary_keyword_hit")
    elif any(keyword in PRIMARY_KEYWORDS for keyword in matched_keywords):
        flags.append("primary_keyword_hit")
    if has_future:
        flags.append("future_tense")
    if has_self_use:
        flags.append("self_use_action")
    if has_outbound:
        flags.append("outbound_or_solution")
    if has_effect:
        flags.append("effect_signal")
    if has_build:
        flags.append("platform_building")
    if has_generic:
        flags.append("generic_system")
    if has_industrial_scene:
        flags.append("industrial_scene")
    if has_background:
        flags.append("background_intro")

    label = None
    confidence = 0.35
    reason = "规则不足以直接判定，建议交给模型补判。"

    if not matched_keywords:
        label = 0
        confidence = 0.98
        reason = "句子未命中关键词词典。"
    elif has_future:
        label = 0
        confidence = 0.92
        reason = "出现未来/规划类表述，按规则倾向未实际应用。"
    elif has_outbound and not has_self_use:
        label = 0
        confidence = 0.88
        reason = "更像对外赋能或解决方案输出，不是企业自身应用。"
    elif has_build and not has_effect and "secondary_keyword_hit" not in flags:
        label = 0
        confidence = 0.82
        reason = "句子更偏平台建设或开发，未体现实际业务使用效果。"
    elif has_generic and not has_industrial_scene:
        label = 0
        confidence = 0.8
        reason = "仅描述通用系统，未提供工业业务场景证据。"
    elif has_background and not has_self_use and not has_effect and "secondary_keyword_hit" not in flags:
        label = 0
        confidence = 0.78
        reason = "更像背景介绍、行业趋势或能力概述，缺少明确落地动作。"
    elif has_self_use and has_effect and has_industrial_scene:
        label = 1
        confidence = 0.9
        reason = "同时具备自身应用动作、工业场景和效果信号，规则判定为已应用。"
    elif secondary_matches:
        confidence = 0.45
        reason = "句子未命中一级关键词，但命中了更宽松的二级关键词，可先纳入候选池等待 AI 进一步判断。"

    return RuleEvaluation(
        matched_keywords=matched_keywords,
        keyword_categories=categories,
        flags=flags,
        label=label,
        confidence=confidence,
        reason=reason,
    )


def contains_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)
