from __future__ import annotations

import re
from dataclasses import dataclass

from report_labeler.keywords import (
    BACKGROUND_WORDS,
    CAPITAL_INVESTMENT_WORDS,
    DISCLOSURE_AUDIT_WORDS,
    EFFECT_WORDS,
    EXTERNAL_SERVICE_WORDS,
    FINANCE_ACCOUNTING_WORDS,
    FUTURE_WORDS,
    GENERIC_SYSTEM_WORDS,
    HIGH_NOISE_PRIMARY_KEYWORDS,
    INDUSTRIAL_SCENE_WORDS,
    OUTBOUND_WORDS,
    PLATFORM_BUILD_WORDS,
    POLICY_REGULATION_WORDS,
    PRIMARY_KEYWORDS_BY_CATEGORY,
    SECONDARY_KEYWORDS_BY_CATEGORY,
    SELF_USE_WORDS,
    TABULAR_NOISE_WORDS,
    WEAK_SECONDARY_KEYWORDS,
)


def normalize_keyword_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；;：、（）()【】\[\]《》“”\"'‘’—\-_/·,.]", "", text)
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

ASCII_KEYWORD_PATTERN = re.compile(r"^[A-Za-z0-9.+_-]+$")
CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
LATIN_CHAR_PATTERN = re.compile(r"[A-Za-z]")


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
        keyword
        for keyword, normalized_keyword in NORMALIZED_PRIMARY_KEYWORDS.items()
        if keyword_matches_sentence(sentence, keyword, normalized_keyword, normalized_sentence)
    ]
    secondary_matches = [
        keyword
        for keyword, normalized_keyword in NORMALIZED_SECONDARY_KEYWORDS.items()
        if keyword not in primary_matches
        and keyword_matches_sentence(sentence, keyword, normalized_keyword, normalized_sentence)
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
    primary_matches = [keyword for keyword in matched_keywords if keyword in PRIMARY_KEYWORDS]
    weak_secondary_matches = [keyword for keyword in secondary_matches if keyword in WEAK_SECONDARY_KEYWORDS]
    high_noise_primary_matches = [keyword for keyword in primary_matches if keyword in HIGH_NOISE_PRIMARY_KEYWORDS]
    flags: list[str] = []

    has_future = contains_any(sentence, FUTURE_WORDS)
    has_self_use = contains_any(sentence, SELF_USE_WORDS)
    has_outbound = contains_any(sentence, OUTBOUND_WORDS)
    has_effect = contains_any(sentence, EFFECT_WORDS)
    has_build = contains_any(sentence, PLATFORM_BUILD_WORDS)
    has_generic = contains_any(sentence, GENERIC_SYSTEM_WORDS)
    has_industrial_scene = contains_any(sentence, INDUSTRIAL_SCENE_WORDS)
    has_background = contains_any(sentence, BACKGROUND_WORDS)
    has_finance = contains_any(sentence, FINANCE_ACCOUNTING_WORDS)
    has_tabular_noise = contains_any(sentence, TABULAR_NOISE_WORDS)
    has_non_chinese_noise = is_non_chinese_noise(sentence)
    has_policy = contains_any(sentence, POLICY_REGULATION_WORDS)
    has_disclosure = contains_any(sentence, DISCLOSURE_AUDIT_WORDS)
    has_external_service = contains_any(sentence, EXTERNAL_SERVICE_WORDS)
    has_capital_investment = contains_any(sentence, CAPITAL_INVESTMENT_WORDS)

    if secondary_matches:
        flags.append("secondary_keyword_hit")
    if primary_matches:
        flags.append("primary_keyword_hit")
    if weak_secondary_matches and not primary_matches and len(weak_secondary_matches) == len(secondary_matches):
        flags.append("weak_secondary_only")
    if high_noise_primary_matches:
        flags.append("high_noise_primary_keyword")
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
    if has_finance:
        flags.append("finance_accounting")
    if has_tabular_noise:
        flags.append("tabular_noise")
    if has_non_chinese_noise:
        flags.append("non_chinese_noise")
    if has_policy:
        flags.append("policy_or_regulation")
    if has_disclosure:
        flags.append("disclosure_or_audit")
    if has_external_service:
        flags.append("external_service")
    if has_capital_investment:
        flags.append("capital_or_investment")

    label = None
    confidence = 0.35
    reason = "规则不足以直接判定，建议交给模型补判。"

    if not matched_keywords:
        label = 0
        confidence = 0.98
        reason = "句子未命中关键词词典。"
    elif has_non_chinese_noise and not has_industrial_scene:
        label = 0
        confidence = 0.97
        reason = "句子更像英文碎片或非主要正文，即使命中关键词也不适合作为候选句。"
    elif has_finance and not has_industrial_scene:
        label = 0
        confidence = 0.96
        reason = "句子主要处于财务或会计披露语境，不属于企业工业互联网应用描述。"
    elif has_disclosure and not (has_industrial_scene and has_self_use):
        label = 0
        confidence = 0.95
        reason = "句子更像审计、会计估值或披露说明，不适合作为企业实际应用候选句。"
    elif has_policy and not has_self_use:
        label = 0
        confidence = 0.94
        reason = "句子主要在转述政策、规划或监管要求，不是企业自身已经落地的应用行为。"
    elif has_tabular_noise and not has_industrial_scene:
        label = 0
        confidence = 0.95
        reason = "句子更像表格、报表或注释片段，不适合作为应用语句候选。"
    elif has_capital_investment and not (has_industrial_scene and has_self_use):
        label = 0
        confidence = 0.93
        reason = "句子更像投资、并购、基金或持股安排，不属于企业内部工业互联网应用。"
    elif has_external_service and not has_industrial_scene:
        label = 0
        confidence = 0.92
        reason = "句子更偏客户服务、供应商管理或园区配套，不是企业自身工业互联网应用。"
    elif high_noise_primary_matches and not (has_industrial_scene or has_self_use or has_effect):
        label = 0
        confidence = 0.95
        reason = "句子命中的主要是一类高噪声关键词，但缺少工业场景、落地动作或效果信号。"
    elif "weak_secondary_only" in flags and not (has_industrial_scene or has_self_use or has_effect):
        label = 0
        confidence = 0.90
        reason = "句子只命中了较弱的二类关键词，且缺少工业场景或应用动作，先不纳入候选。"
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
        confidence = 0.80
        reason = "仅描述通用系统，未提供工业业务场景证据。"
    elif has_background and not has_self_use and not has_effect and "secondary_keyword_hit" not in flags:
        label = 0
        confidence = 0.78
        reason = "更像背景介绍、行业趋势或能力概述，缺少明确落地动作。"
    elif has_self_use and has_effect and has_industrial_scene:
        label = 1
        confidence = 0.90
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


def keyword_matches_sentence(
    sentence: str,
    keyword: str,
    normalized_keyword: str,
    normalized_sentence: str,
) -> bool:
    if not normalized_keyword:
        return False
    if ASCII_KEYWORD_PATTERN.fullmatch(keyword):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])", re.IGNORECASE)
        return bool(pattern.search(sentence))
    return normalized_keyword in normalized_sentence


def is_non_chinese_noise(sentence: str) -> bool:
    chinese_count = len(CHINESE_CHAR_PATTERN.findall(sentence))
    latin_count = len(LATIN_CHAR_PATTERN.findall(sentence))
    return chinese_count <= 2 and latin_count >= 8
