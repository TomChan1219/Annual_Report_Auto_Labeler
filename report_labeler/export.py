from __future__ import annotations

from io import BytesIO

import pandas as pd

from report_labeler.models import JudgmentRecord


def build_submission_dataframe(records: list[JudgmentRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        label = record.reviewed_label if record.reviewed_label is not None else record.final_label
        rows.append(
            {
                "句子内容": record.sentence,
                "来源文件": file_name_only(record.source_file),
                "人工标注标签": label,
                "id": record.stock_id,
                "year": record.year,
                "判断理由": record.judge_reason,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["句子内容", "来源文件", "人工标注标签", "id", "year", "判断理由"],
    )


def build_analysis_dataframe(records: list[JudgmentRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "record_id": record.record_id,
                "来源文件": file_name_only(record.source_file),
                "source_file": record.source_file,
                "company_name": record.company_name,
                "id": record.stock_id,
                "year": record.year,
                "句子内容": record.sentence,
                "sentence_index": record.sentence_index,
                "char_position": record.char_position,
                "matched_keywords": ", ".join(record.matched_keywords),
                "keyword_categories": ", ".join(record.keyword_categories),
                "rule_flags": ", ".join(record.rule_flags),
                "rule_label": record.rule_label,
                "model_label": record.model_label,
                "final_label": record.final_label,
                "reviewed_label": record.reviewed_label,
                "confidence": record.confidence,
                "判断理由": record.judge_reason,
                "context_before": record.context_before,
                "context_after": record.context_after,
                "model_source": record.model_source,
                "review_note": record.review_note,
            }
        )
    return pd.DataFrame(rows)


def export_submission_xlsx(records: list[JudgmentRecord], output_path: str | None = None) -> bytes:
    return write_excel_bytes({"提交简表": build_submission_dataframe(records)}, output_path)


def export_analysis_xlsx(records: list[JudgmentRecord], output_path: str | None = None) -> bytes:
    return write_excel_bytes({"分析详表": build_analysis_dataframe(records)}, output_path)


def export_dual_xlsx(records: list[JudgmentRecord], output_path: str | None = None) -> bytes:
    sheets = {
        "提交简表": build_submission_dataframe(records),
        "分析详表": build_analysis_dataframe(records),
    }
    return write_excel_bytes(sheets, output_path)


def write_excel_bytes(sheets: dict[str, pd.DataFrame], output_path: str | None = None) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    payload = buffer.getvalue()
    if output_path:
        with open(output_path, "wb") as fh:
            fh.write(payload)
    return payload


def file_name_only(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]
