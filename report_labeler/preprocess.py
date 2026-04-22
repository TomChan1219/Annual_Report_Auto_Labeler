from __future__ import annotations

import re


PAGE_NOISE_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^第.{1,6}节"),
    re.compile(r"^目录$"),
    re.compile(r"^释义$"),
    re.compile(r"^年度报告$"),
]

TITLE_PREFIX_PATTERN = re.compile(
    r"^\s*[一二三四五六七八九十0-9]+[、.．)]\s*[^\s。！？；;:：]{1,30}\s+"
)

LEADING_FRAGMENT_MARKERS = re.compile(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*")
LEADING_FRAGMENT_REWRITES = {
    "模式公司": "公司",
    "年12月": "12月",
}
FRAGMENT_START_PREFIXES = (
    "款的",
    "自筹资金",
    "对于授予的",
    "指通过",
    "本公司之子公司",
    "5，",
)
FRAGMENT_END_SUFFIXES = (
    "融资租赁",
    "信息平台",
    "系统中",
    "平台",
    "建设",
    "其中",
    "以及",
    "并",
    "与",
    "款",
    "商",
    "清",
)


def clean_text(text: str) -> str:
    lines = []
    for raw_line in text.replace("\ufeff", "").splitlines():
        line = normalize_whitespace(raw_line)
        if not line:
            continue
        if any(pattern.match(line) for pattern in PAGE_NOISE_PATTERNS):
            continue
        if len(line) <= 2 and line.isdigit():
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_sentence_text(text: str) -> str:
    text = normalize_whitespace(text)
    # Remove stray spaces inserted between Chinese characters by OCR/PDF text extraction.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    # Remove spaces between Chinese characters and punctuation.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。！？；：、）】》])", "", text)
    text = re.sub(r"(?<=[（【《])\s+(?=[\u4e00-\u9fff])", "", text)
    # Remove spaces around common punctuation while keeping English word spacing intact.
    text = re.sub(r"\s+([，。！？；：、）】》])", r"\1", text)
    text = re.sub(r"([（【《])\s+", r"\1", text)
    text = strip_fragment_noise(text)
    return text.strip()


def normalize_sentence_key(text: str) -> str:
    normalized = normalize_sentence_text(text)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[，,。！？；;：:、（）()【】\[\]《》“”\"'‘’\-—_/·]", "", normalized)
    return normalized


def is_title_like(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized:
        return True

    if normalized.endswith(("：", ":")) and len(normalized) <= 30:
        return True

    if re.fullmatch(r"[一二三四五六七八九十0-9]+[、.．)]?.{0,20}", normalized):
        return True

    if len(normalized) <= 18 and not re.search(r"[。！？；;，,:：]", normalized):
        return True

    if re.fullmatch(r"[A-Za-z0-9一-龥\-_/（）()【】\[\]·]+", normalized) and len(normalized) <= 24:
        return True

    return False


def split_sentences(text: str, min_chars: int = 8) -> list[str]:
    sentences: list[str] = []
    buffer = ""
    for raw_line in text.splitlines():
        token = normalize_whitespace(raw_line)
        if not token:
            continue
        if buffer and should_break_before_line(buffer, token):
            current = normalize_whitespace(buffer)
            if len(current) >= min_chars:
                sentences.append(current)
            buffer = ""
        buffer = normalize_whitespace(f"{buffer} {token}" if buffer else token)
        if ends_with_sentence_punctuation(buffer):
            current = normalize_whitespace(buffer)
            buffer = ""
            if len(current) >= min_chars:
                sentences.append(current)
    if buffer and len(buffer) >= min_chars:
        sentences.append(normalize_whitespace(buffer))
    normalized_sentences = []
    for sentence in sentences:
        stripped = normalize_sentence_text(strip_title_prefix(sentence))
        if stripped and len(stripped) >= min_chars:
            normalized_sentences.append(stripped)
    merged_sentences = merge_fragmented_sentences(normalized_sentences, min_chars=min_chars)
    return dedupe_preserve_order(merged_sentences)


def should_break_before_line(current_buffer: str, next_line: str) -> bool:
    current = normalize_whitespace(current_buffer)
    following = normalize_whitespace(next_line)
    if not current or not following:
        return False
    if ends_with_sentence_punctuation(current):
        return True
    if is_title_like(following):
        return True
    if is_table_like(following):
        return True
    if re.match(r"^\(?[0-9一二三四五六七八九十]+\)?[、.．)]", following):
        return True
    return False


def ends_with_sentence_punctuation(text: str) -> bool:
    normalized = normalize_whitespace(text)
    return normalized.endswith(("。", "！", "？", "；", ";"))


def merge_fragmented_sentences(sentences: list[str], min_chars: int = 8) -> list[str]:
    if not sentences:
        return []

    merged: list[str] = []
    idx = 0
    while idx < len(sentences):
        current = normalize_sentence_text(sentences[idx])
        while idx + 1 < len(sentences) and should_merge_with_next(current, sentences[idx + 1]):
            current = normalize_sentence_text(f"{current}{sentences[idx + 1]}")
            idx += 1
        if current and len(current) >= min_chars:
            merged.append(current)
        idx += 1
    return merged


def should_merge_with_next(current: str, next_sentence: str) -> bool:
    current_text = normalize_sentence_text(current)
    next_text = normalize_sentence_text(next_sentence)
    if not current_text or not next_text:
        return False
    if ends_with_sentence_punctuation(current_text):
        return False
    if len(current_text) >= 220:
        return False

    bad_next_prefixes = (
        "款的",
        "自筹资金",
        "模式公司",
        "对于授予的",
        "本公司之子公司",
        "指通过",
        "其中",
        "以及",
        "并",
        "同时",
        "商",
        "清",
    )
    continuation_start = (
        next_text.startswith(bad_next_prefixes)
        or re.match(r"^[a-zA-Z0-9%.,，、）)]", next_text) is not None
    )
    current_fragment = current_text.endswith(
        FRAGMENT_END_SUFFIXES
    )
    return continuation_start or current_fragment


def strip_fragment_noise(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = LEADING_FRAGMENT_MARKERS.sub("", cleaned, count=1)
    for source, target in LEADING_FRAGMENT_REWRITES.items():
        if cleaned.startswith(source):
            cleaned = target + cleaned[len(source) :]
            break
    return cleaned.strip()


def is_fragment_like(text: str) -> bool:
    cleaned = normalize_sentence_text(text)
    if not cleaned:
        return True
    if cleaned.startswith(FRAGMENT_START_PREFIXES):
        return True
    if not ends_with_sentence_punctuation(cleaned) and cleaned.endswith(FRAGMENT_END_SUFFIXES):
        return True
    return False


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def strip_title_prefix(text: str) -> str:
    normalized = normalize_whitespace(text)
    stripped = TITLE_PREFIX_PATTERN.sub("", normalized, count=1)
    if stripped and len(stripped) >= max(8, len(normalized) // 3):
        return stripped
    return normalized


def is_table_like(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized:
        return True
    digit_count = sum(ch.isdigit() for ch in normalized)
    ratio = digit_count / max(len(normalized), 1)
    if digit_count >= 8 and ratio >= 0.18:
        return True
    if normalized.count("□") + normalized.count("√") >= 2:
        return True
    return False
