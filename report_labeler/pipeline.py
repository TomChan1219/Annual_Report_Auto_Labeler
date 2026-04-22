from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from typing import Callable

from report_labeler.io_utils import build_document_record, detect_and_read_text, list_txt_files
from report_labeler.llm import get_judge, merge_judgment
from report_labeler.models import (
    JudgmentRecord,
    ModelConfig,
    PipelineConfig,
    ProcessingError,
    SentenceRecord,
)
from report_labeler.preprocess import clean_text, is_title_like, split_sentences
from report_labeler.rules import evaluate_rules


ProgressCallback = Callable[[str, int, int, str], None]


def run_single(
    file_path: str,
    model_config: ModelConfig,
    pipeline_config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, list]:
    preview = preview_single(file_path, pipeline_config, progress_callback=progress_callback)
    return judge_preview(preview, model_config, progress_callback=progress_callback)


def run_batch(
    folder_path: str,
    model_config: ModelConfig,
    pipeline_config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, list]:
    file_paths = list_txt_files(folder_path, pipeline_config.max_files)
    preview = preview_files(file_paths, pipeline_config, progress_callback=progress_callback)
    return judge_preview(preview, model_config, progress_callback=progress_callback)


def preview_single(
    file_path: str,
    pipeline_config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, list]:
    return preview_files([file_path], pipeline_config, progress_callback=progress_callback)


def preview_files(
    file_paths: list[str],
    pipeline_config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, list]:
    documents = []
    raw_sentence_records: list[SentenceRecord] = []
    errors: list[ProcessingError] = []
    recall_counter: Counter[str] = Counter()

    total_files = len(file_paths)
    for index, file_path in enumerate(file_paths, start=1):
        notify(progress_callback, "read", index - 1, total_files, f"正在读取 {os.path.basename(file_path)}")
        try:
            raw_text, encoding, warnings = detect_and_read_text(file_path, pipeline_config.encodings)
            cleaned = clean_text(raw_text)
            if len(cleaned.strip()) < pipeline_config.min_sentence_chars:
                errors.append(ProcessingError(file_path, "read", "文件内容过短或为空"))
                continue
            document = build_document_record(file_path, cleaned, raw_text, encoding, warnings)
            documents.append(document)
            sentences = split_sentences(cleaned, min_chars=pipeline_config.min_sentence_chars)
            candidates = build_sentence_records(document, sentences, pipeline_config)
            raw_sentence_records.extend(candidates)
            recall_counter[file_path] += len(candidates)
        except Exception as exc:  # noqa: BLE001
            errors.append(ProcessingError(file_path, "preprocess", str(exc)))
        finally:
            notify(progress_callback, "read", index, total_files, f"已完成读取 {index}/{total_files}")

    selected_sentence_records, selection_meta = select_candidates(
        raw_sentence_records,
        [doc.source_file for doc in documents],
        pipeline_config,
    )
    selection_meta["document_details"] = {
        doc.source_file: {"encoding": doc.encoding, "warnings": doc.warnings}
        for doc in documents
    }

    summary = build_summary(
        judgments=[],
        document_paths=[doc.source_file for doc in documents],
        errors=errors,
        recall_counter=recall_counter,
        selection_meta=selection_meta,
        pipeline_config=pipeline_config,
        selected_count=len(selected_sentence_records),
    )
    return {
        "documents": documents,
        "sentences": selected_sentence_records,
        "judgments": [],
        "errors": errors,
        "summary": summary,
        "stage": "preview",
    }


def judge_preview(
    preview_result: dict[str, list],
    model_config: ModelConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, list]:
    judge = get_judge(model_config)
    sentence_records: list[SentenceRecord] = preview_result["sentences"]
    judgments: list[JudgmentRecord] = []

    total_sentences = len(sentence_records)
    for index, sentence_record in enumerate(sentence_records, start=1):
        notify(
            progress_callback,
            "judge",
            index - 1,
            total_sentences,
            f"正在判断第 {index}/{max(total_sentences, 1)} 条候选句",
        )
        try:
            model_result = judge.judge(sentence_record, model_config)
            judgments.append(merge_judgment(sentence_record, model_result))
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"这句话暂时回退为规则结果，原因是模型调用失败：{exc}"
            judgments.append(
                JudgmentRecord(
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
                    model_label=None,
                    final_label=sentence_record.rule_label if sentence_record.rule_label is not None else 0,
                    confidence=sentence_record.rule_confidence,
                    judge_reason=fallback_reason,
                    model_source="rules_fallback",
                )
            )
        finally:
            notify(progress_callback, "judge", index, total_sentences, "判断阶段进行中")

    result = dict(preview_result)
    result["judgments"] = judgments
    result["stage"] = "judged"
    result["summary"] = build_summary(
        judgments=judgments,
        document_paths=[doc.source_file for doc in preview_result["documents"]],
        errors=preview_result["errors"],
        recall_counter=Counter(),
        selection_meta={
            "max_total_allowed": preview_result["summary"]["max_total_allowed"],
            "dynamic_target_per_file": preview_result["summary"]["dynamic_target_per_file"],
            "files_below_target": preview_result["summary"]["files_below_target"],
            "files_above_target": preview_result["summary"]["files_above_target"],
            "selected_counter": Counter(item.source_file for item in sentence_records),
            "document_details": {
                doc.source_file: {"encoding": doc.encoding, "warnings": doc.warnings}
                for doc in preview_result["documents"]
            },
        },
        pipeline_config=PipelineConfig(
            target_sentences_per_file=preview_result["summary"]["target_sentences_per_file"],
            min_total_sentences=preview_result["summary"]["min_total_sentences"],
        ),
        selected_count=len(sentence_records),
    )
    result["summary"]["raw_recall_total"] = preview_result["summary"]["raw_recall_total"]
    return result


def build_sentence_records(document, sentences: list[str], pipeline_config: PipelineConfig) -> list[SentenceRecord]:
    records: list[SentenceRecord] = []
    context_window = pipeline_config.context_window
    seen_sentences: set[str] = set()
    secondary_enabled = allows_secondary_keywords(pipeline_config.filter_strength)

    for idx, sentence in enumerate(sentences):
        normalized_sentence = sentence.strip()
        if normalized_sentence in seen_sentences:
            continue
        seen_sentences.add(normalized_sentence)

        if is_title_like(normalized_sentence):
            continue

        evaluation = evaluate_rules(normalized_sentence)
        if not evaluation.matched_keywords:
            continue
        if not secondary_enabled and "primary_keyword_hit" not in evaluation.flags:
            continue

        stored_sentence = normalized_sentence
        if len(stored_sentence) > pipeline_config.sentence_max_chars:
            stored_sentence = stored_sentence[: pipeline_config.sentence_max_chars].rstrip() + "..."

        context_before = " ".join(sentences[max(0, idx - context_window) : idx])
        context_after = " ".join(sentences[idx + 1 : idx + 1 + context_window])
        char_position = document.cleaned_text.find(normalized_sentence[:50])

        records.append(
            SentenceRecord(
                record_id=f"{os.path.basename(document.source_file)}::{idx}",
                source_file=document.source_file,
                stock_id=document.stock_id,
                year=document.year,
                company_name=document.company_name,
                sentence=stored_sentence,
                sentence_index=idx,
                char_position=char_position,
                context_before=context_before,
                context_after=context_after,
                matched_keywords=evaluation.matched_keywords,
                keyword_categories=evaluation.keyword_categories,
                rule_flags=evaluation.flags,
                rule_label=evaluation.label,
                rule_confidence=evaluation.confidence,
                rule_reason=evaluation.reason,
            )
        )

    return records


def select_candidates(
    sentence_records: list[SentenceRecord],
    document_paths: list[str],
    pipeline_config: PipelineConfig,
) -> tuple[list[SentenceRecord], dict]:
    grouped: dict[str, list[tuple[float, SentenceRecord]]] = defaultdict(list)
    for record in sentence_records:
        grouped[record.source_file].append((score_candidate(record, pipeline_config.filter_strength), record))

    for file_path in grouped:
        grouped[file_path].sort(key=lambda item: sort_key(item[0], item[1]), reverse=True)

    document_count = max(len(document_paths), 1)
    target_total = pipeline_config.min_total_sentences
    max_total_allowed = max(
        target_total,
        int(math.ceil(target_total * (1 + pipeline_config.max_overage_ratio))),
    )
    dynamic_target_per_file = max(
        1,
        min(
            pipeline_config.target_sentences_per_file,
            int(math.ceil(target_total / document_count)),
        ),
    )

    selected: list[SentenceRecord] = []
    selected_ids: set[str] = set()
    selection_counter: Counter[str] = Counter()

    for file_path in document_paths:
        ranked = grouped.get(file_path, [])
        for _, record in ranked[:dynamic_target_per_file]:
            selected.append(record)
            selected_ids.add(record.record_id)
            selection_counter[file_path] += 1

    if len(selected) < target_total:
        remaining = build_remaining_pool(grouped, selected_ids)
        remaining.sort(key=lambda item: sort_key(item[0], item[1]), reverse=True)
        while remaining and len(selected) < target_total:
            _, record = remaining.pop(0)
            selected.append(record)
            selected_ids.add(record.record_id)
            selection_counter[record.source_file] += 1

    if len(selected) > max_total_allowed:
        selected.sort(key=lambda item: sort_key(score_candidate(item, pipeline_config.filter_strength), item), reverse=True)
        selected = selected[:max_total_allowed]
        selection_counter = Counter(item.source_file for item in selected)

    files_below_target = []
    files_above_target = []
    for file_path in document_paths:
        count = selection_counter.get(file_path, 0)
        if count < pipeline_config.target_sentences_per_file:
            files_below_target.append(
                {
                    "source_file": file_path,
                    "selected_count": count,
                    "target": pipeline_config.target_sentences_per_file,
                }
            )
        if count > pipeline_config.target_sentences_per_file:
            files_above_target.append(
                {
                    "source_file": file_path,
                    "selected_count": count,
                    "target": pipeline_config.target_sentences_per_file,
                }
            )

    meta = {
        "target_total": target_total,
        "max_total_allowed": max_total_allowed,
        "dynamic_target_per_file": dynamic_target_per_file,
        "selected_counter": selection_counter,
        "files_below_target": files_below_target,
        "files_above_target": files_above_target,
    }
    return selected, meta


def build_remaining_pool(grouped, selected_ids: set[str]) -> list[tuple[float, SentenceRecord]]:
    remaining: list[tuple[float, SentenceRecord]] = []
    for ranked in grouped.values():
        for score, record in ranked:
            if record.record_id not in selected_ids:
                remaining.append((score, record))
    return remaining


def score_candidate(record: SentenceRecord, filter_strength: str) -> float:
    score = float(len(record.matched_keywords))
    flags = set(record.rule_flags)
    mode = normalize_filter_mode(filter_strength)
    secondary_enabled = allows_secondary_keywords(mode)

    if "primary_keyword_hit" in flags:
        score += 4.0
    if "secondary_keyword_hit" in flags and secondary_enabled:
        score += 1.5 if mode == "二类关键词兜底" else 0.5

    if record.rule_label == 1:
        score += 5.0
    if record.rule_label == 0:
        score -= 0.2 if mode == "二类关键词兜底" and "primary_keyword_hit" in flags else (
            0.5 if mode == "二类关键词兜底" else 1.5
        )
    if "self_use_action" in flags:
        score += 3.0
    if "industrial_scene" in flags:
        score += 2.5
    if "effect_signal" in flags:
        score += 2.0
    if "future_tense" in flags:
        score -= 1.0 if mode == "二类关键词兜底" else 3.5
    if "outbound_or_solution" in flags:
        score -= 1.0 if mode == "二类关键词兜底" else 3.0
    if "platform_building" in flags:
        score -= 0.8 if mode == "二类关键词兜底" else 2.0
    if "generic_system" in flags:
        score -= 0.5 if mode == "二类关键词兜底" else 1.5
    if "background_intro" in flags:
        score -= 0.5 if mode == "二类关键词兜底" else 2.5
    score += min(record.rule_confidence, 1.0)
    return score


def sort_key(score: float, record: SentenceRecord) -> tuple[float, float, int]:
    return (score, record.rule_confidence, -record.sentence_index)


def build_summary(
    judgments: list[JudgmentRecord],
    document_paths: list[str],
    errors: list[ProcessingError],
    recall_counter: Counter[str],
    selection_meta: dict,
    pipeline_config: PipelineConfig,
    selected_count: int,
) -> dict:
    selection_counter: Counter[str] = selection_meta.get("selected_counter", Counter())
    document_details = selection_meta.get("document_details", {})
    documents_without_candidates = [
        path for path in document_paths if selection_counter.get(path, 0) == 0
    ]
    file_stats = []
    for path in document_paths:
        details = document_details.get(path, {})
        file_stats.append(
            {
                "source_file": path,
                "raw_recall_count": recall_counter.get(path, 0),
                "selected_count": selection_counter.get(path, 0),
                "target": pipeline_config.target_sentences_per_file,
                "encoding": details.get("encoding"),
                "warnings": ", ".join(details.get("warnings", [])),
            }
        )
    return {
        "document_count": len(document_paths),
        "candidate_count": selected_count,
        "positive_count": sum(1 for item in judgments if item.final_label == 1),
        "negative_count": sum(1 for item in judgments if item.final_label == 0),
        "error_count": len(errors),
        "meets_total_target": selected_count >= pipeline_config.min_total_sentences,
        "target_sentences_per_file": pipeline_config.target_sentences_per_file,
        "min_total_sentences": pipeline_config.min_total_sentences,
        "max_total_allowed": selection_meta["max_total_allowed"],
        "dynamic_target_per_file": selection_meta["dynamic_target_per_file"],
        "raw_recall_total": sum(recall_counter.values()),
        "files_below_target": selection_meta["files_below_target"],
        "files_above_target": selection_meta["files_above_target"],
        "documents_without_candidates": documents_without_candidates,
        "file_stats": file_stats,
    }


def notify(progress_callback: ProgressCallback | None, stage: str, current: int, total: int, message: str) -> None:
    if progress_callback:
        progress_callback(stage, current, total, message)


def normalize_filter_mode(filter_strength: str | None) -> str:
    if filter_strength in {"二类关键词兜底", "宽松", "平衡"}:
        return "二类关键词兜底"
    return "一类关键词"


def allows_secondary_keywords(filter_strength: str | None) -> bool:
    return normalize_filter_mode(filter_strength) == "二类关键词兜底"
