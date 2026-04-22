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
    r"^\s*[一二三四五六七八九十0-9]+[、.．)]\s*[^。！？；;\n]{0,28}(?:显著|先机|规划|布局|质量|能力|水平|建设|管理)?\s+"
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
    merged = text.replace("\n", "\n")
    parts = re.split(r"(?<=[。！？；;])|(?<=\n)", merged)
    sentences: list[str] = []
    buffer = ""
    for part in parts:
        token = normalize_whitespace(part)
        if not token:
            continue
        if token.endswith(("。", "！", "？", "；", ";")):
            current = normalize_whitespace(buffer + " " + token)
            buffer = ""
            if len(current) >= min_chars:
                sentences.append(current)
        else:
            buffer = normalize_whitespace(buffer + " " + token)
            if len(buffer) >= 80:
                sentences.append(buffer)
                buffer = ""
    if buffer and len(buffer) >= min_chars:
        sentences.append(buffer)
    normalized_sentences = []
    for sentence in sentences:
        stripped = strip_title_prefix(sentence)
        if stripped and len(stripped) >= min_chars:
            normalized_sentences.append(stripped)
    return dedupe_preserve_order(normalized_sentences)


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
