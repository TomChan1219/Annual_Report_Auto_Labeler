from __future__ import annotations

import os
import re
from pathlib import Path

from report_labeler.models import DocumentRecord


FILENAME_PATTERN = re.compile(
    r"^(?P<stock_id>\d{6}|\d{5,6})_(?P<year>\d{4})_(?P<company_name>.+?)_(?P<report_type>.+?)_(?P<report_date>\d{4}-\d{2}-\d{2})\.txt$"
)


def list_txt_files(path: str, max_files: int | None = None) -> list[str]:
    input_path = Path(path)
    files: list[str] = []
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        files = [str(input_path.resolve())]
    elif input_path.is_dir():
        files = sorted(str(p.resolve()) for p in input_path.rglob("*.txt"))
    if max_files is not None:
        files = files[:max_files]
    return files


def detect_and_read_text(file_path: str, encodings: list[str]) -> tuple[str, str, list[str]]:
    raw = Path(file_path).read_bytes()
    warnings: list[str] = []
    if not raw:
        return "", encodings[0] if encodings else "utf-8", ["empty_file"]

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
        return text, "utf-16", warnings

    candidates: list[tuple[float, str, str, list[str]]] = []
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            local_warnings: list[str] = []
            if "\x00" in text and encoding != "utf-16":
                local_warnings.append(f"suspicious_nulls:{encoding}")
            score = score_decoded_text(text, encoding)
            candidates.append((score, text, encoding, local_warnings))
        except UnicodeDecodeError:
            continue

    if candidates:
        score, text, encoding, local_warnings = max(
            candidates,
            key=lambda item: (item[0], encoding_priority(item[2])),
        )
        if encoding == "utf-8-sig" and not raw.startswith(b"\xef\xbb\xbf"):
            encoding = "utf-8"
        warnings.extend(local_warnings)
        if score < 0.35:
            warnings.append(f"low_decode_confidence:{encoding}")
        return text, encoding, warnings

    text = raw.decode(encodings[0] if encodings else "utf-8", errors="ignore")
    warnings.append("decode_fallback_ignore_errors")
    return text, encodings[0] if encodings else "utf-8", warnings


def score_decoded_text(text: str, encoding: str) -> float:
    sample = text[:4000]
    if not sample:
        return 0.0

    total = len(sample)
    printable_ratio = sum(ch.isprintable() or ch in "\r\n\t" for ch in sample) / total
    cjk_ratio = sum("\u4e00" <= ch <= "\u9fff" for ch in sample) / total
    ascii_ratio = sum(ch.isascii() and (ch.isalnum() or ch in " \r\n\t:：-_/().,%") for ch in sample) / total
    private_ratio = sum("\ue000" <= ch <= "\uf8ff" for ch in sample) / total
    replacement_ratio = sample.count("\ufffd") / total
    control_ratio = sum((ord(ch) < 32 and ch not in "\r\n\t") for ch in sample) / total

    score = printable_ratio + 0.6 * cjk_ratio + 0.2 * ascii_ratio
    score -= 2.0 * private_ratio
    score -= 1.5 * replacement_ratio
    score -= 1.5 * control_ratio

    if encoding == "utf-16" and not text.startswith("\ufeff") and cjk_ratio == 0 and ascii_ratio < 0.2:
        score -= 0.5

    return score


def encoding_priority(encoding: str) -> int:
    priorities = {
        "utf-8-sig": 5,
        "utf-8": 4,
        "gb18030": 3,
        "gbk": 2,
        "utf-16": 1,
    }
    return priorities.get(encoding, 0)


def parse_filename(file_path: str) -> dict[str, str | None]:
    name = os.path.basename(file_path)
    match = FILENAME_PATTERN.match(name)
    if not match:
        return {
            "stock_id": None,
            "year": None,
            "company_name": os.path.splitext(name)[0],
            "report_date": None,
        }
    data = match.groupdict()
    return {
        "stock_id": data["stock_id"],
        "year": data["year"],
        "company_name": data["company_name"],
        "report_date": data["report_date"],
    }


def build_document_record(
    file_path: str,
    cleaned_text: str,
    raw_text: str,
    encoding: str,
    warnings: list[str],
) -> DocumentRecord:
    parsed = parse_filename(file_path)
    if parsed["stock_id"] is None:
        warnings = warnings + ["filename_parse_failed"]
    return DocumentRecord(
        source_file=file_path,
        stock_id=parsed["stock_id"],
        year=parsed["year"],
        company_name=parsed["company_name"],
        report_date=parsed["report_date"],
        encoding=encoding,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        warnings=warnings,
    )
