from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PipelineConfig:
    encodings: list[str] = field(
        default_factory=lambda: ["utf-16", "utf-8-sig", "utf-8", "gb18030", "gbk"]
    )
    max_files: int | None = None
    sentence_max_chars: int = 220
    context_window: int = 1
    min_sentence_chars: int = 8
    target_sentences_per_file: int = 15
    min_total_sentences: int = 180
    max_overage_ratio: float = 0.2
    filter_strength: str = "二类关键词兜底"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    provider: str = "mock"
    model_name: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    timeout_seconds: int = 60
    concurrency: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentRecord:
    source_file: str
    stock_id: str | None
    year: str | None
    company_name: str | None
    report_date: str | None
    encoding: str | None
    raw_text: str
    cleaned_text: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class SentenceRecord:
    record_id: str
    source_file: str
    stock_id: str | None
    year: str | None
    company_name: str | None
    sentence: str
    sentence_index: int
    char_position: int
    context_before: str
    context_after: str
    matched_keywords: list[str]
    keyword_categories: list[str]
    rule_flags: list[str]
    rule_label: int | None
    rule_confidence: float
    rule_reason: str


@dataclass
class JudgmentRecord:
    record_id: str
    source_file: str
    stock_id: str | None
    year: str | None
    company_name: str | None
    sentence: str
    sentence_index: int
    char_position: int
    context_before: str
    context_after: str
    matched_keywords: list[str]
    keyword_categories: list[str]
    rule_flags: list[str]
    rule_label: int | None
    model_label: int | None
    final_label: int
    confidence: float
    judge_reason: str
    model_source: str
    reviewed_label: int | None = None
    review_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingError:
    source_file: str
    stage: str
    message: str
