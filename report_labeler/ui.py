from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from report_labeler.export import build_analysis_dataframe, build_preview_export_dataframe, export_dual_xlsx, export_preview_xlsx
from report_labeler.io_utils import parse_filename
from report_labeler.models import JudgmentRecord, ModelConfig, PipelineConfig
from report_labeler.pipeline import judge_preview, preview_files


def main() -> None:
    st.set_page_config(page_title="企业年报工业互联网应用自动识别工具", layout="wide")
    st.title("企业年报工业互联网应用自动识别工具")
    st.caption("先预览候选句，再确认后调用 AI 标注；如需更宽松召回，可适当放宽关键词策略后再人工筛选。")

    init_state()

    with st.sidebar:
        st.header("运行配置")
        root_folder = st.text_input(
            "年报总文件夹",
            value=st.session_state.get("root_folder", ""),
            help="先设置存放所有年报 txt 的总文件夹，后续只输入文件名。",
        )
        run_mode = st.radio("处理模式", ["单文件", "批量文件名列表"], index=0)
        file_names_text = st.text_area(
            "年报文件名",
            value=st.session_state.get("file_names_text", ""),
            height=140,
            help="单文件模式输入一个文件名；批量模式一行一个文件名。",
            placeholder="600461_2020_洪城水业_江西洪城环境股份有限公司2020年年度报告_2021-04-21.txt",
        )
        output_path = st.text_input(
            "导出文件路径",
            value=st.session_state.get("output_path", str(Path.cwd() / "outputs" / "annual_report_results.xlsx")),
        )
        provider = st.selectbox("判断模式", ["mock", "deepseek"], index=0)
        model_name = st.text_input("模型名称", value="deepseek-chat")
        base_url = st.text_input("Base URL", value="https://api.deepseek.com")
        api_key = st.text_input("DeepSeek API Key", value="", type="password")
        target_per_file = st.number_input("每篇期望句子数", min_value=1, value=15, step=1)
        min_total_sentences = st.number_input("总最少句子数", min_value=1, value=180, step=1)
        max_overage_percent = st.slider("总句子允许超额比例", min_value=0, max_value=100, value=20, step=5)
        filter_strength = st.selectbox(
            "关键词召回策略",
            ["一类关键词", "二类关键词兜底"],
            index=1,
            help="一类关键词：只保留文档中命中作业原始词典的句子。二类关键词兜底：在一类关键词基础上，额外允许更宽松的兜底关键词，尽量保证每篇年报都有一定句子数。",
        )
        st.caption("一类关键词：只出现在文档里的作业原始关键词。二类关键词：为保证每篇年报有一定句子数而加入的兜底关键词。")
        sentence_max_chars = st.number_input("单句最大展示长度", min_value=80, value=220, step=10)
        context_window = st.number_input("上下文窗口句数", min_value=0, value=1, step=1)
        preview_clicked = st.button("1. 先预览候选句", type="primary", use_container_width=True)
        judge_clicked = st.button("2. 确认后开始 AI 标注", use_container_width=True)

    tabs = st.tabs(["1. 运行", "2. 概览", "3. 审核", "4. 导出"])

    if preview_clicked:
        st.session_state.root_folder = root_folder
        st.session_state.file_names_text = file_names_text
        st.session_state.output_path = output_path
        execute_preview(
            root_folder=root_folder,
            run_mode=run_mode,
            file_names_text=file_names_text,
            pipeline_config=PipelineConfig(
                sentence_max_chars=sentence_max_chars,
                context_window=context_window,
                target_sentences_per_file=target_per_file,
                min_total_sentences=min_total_sentences,
                max_overage_ratio=max_overage_percent / 100,
                filter_strength=filter_strength,
            ),
        )

    if judge_clicked:
        execute_judging(
            model_config=ModelConfig(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
            )
        )

    with tabs[0]:
        render_run_summary()
    with tabs[1]:
        render_overview()
    with tabs[2]:
        render_review()
    with tabs[3]:
        render_export(output_path)


def init_state() -> None:
    st.session_state.setdefault("run_result", None)
    st.session_state.setdefault("edited_analysis_df", None)
    st.session_state.setdefault("excluded_files", [])
    st.session_state.setdefault("export_payload", None)
    st.session_state.setdefault("export_record_ids", [])


def execute_preview(root_folder: str, run_mode: str, file_names_text: str, pipeline_config: PipelineConfig) -> None:
    file_paths = resolve_inputs(root_folder, run_mode, file_names_text)
    if file_paths is None:
        return

    read_progress = st.progress(0, text="准备开始")
    status_box = st.empty()

    def progress_callback(stage: str, current: int, total: int, message: str) -> None:
        if stage != "read":
            return
        ratio = 0.0 if total <= 0 else current / total
        read_progress.progress(min(ratio, 1.0), text=f"读取阶段：{message}")
        status_box.info(message)

    with st.spinner("正在筛选候选句，请稍候..."):
        result = preview_files(file_paths, pipeline_config, progress_callback=progress_callback)

    st.session_state.run_result = result
    st.session_state.edited_analysis_df = None
    st.session_state.excluded_files = []
    st.session_state.export_payload = None
    st.session_state.export_record_ids = []
    st.success(f"预览完成，共筛出 {len(result['sentences'])} 条候选句。请先检查数量和文件分布，再决定是否调用 AI。")


def execute_judging(model_config: ModelConfig) -> None:
    preview_result = st.session_state.get("run_result")
    if not preview_result or preview_result.get("stage") != "preview":
        st.error("请先完成候选句预览，再开始 AI 标注。")
        return

    judge_progress = st.progress(0, text="等待开始")
    status_box = st.empty()

    def progress_callback(stage: str, current: int, total: int, message: str) -> None:
        if stage != "judge":
            return
        ratio = 0.0 if total <= 0 else current / total
        judge_progress.progress(min(ratio, 1.0), text=f"判断阶段：{message}")
        status_box.info(message)

    with st.spinner("正在调用 AI 标注，请稍候..."):
        result = judge_preview(preview_result, model_config, progress_callback=progress_callback)

    st.session_state.run_result = result
    st.session_state.edited_analysis_df = build_analysis_dataframe(result["judgments"])
    st.session_state.export_payload = None
    st.session_state.export_record_ids = []
    st.success(f"AI 标注完成，共处理 {len(result['judgments'])} 条候选句。")


def resolve_inputs(root_folder: str, run_mode: str, file_names_text: str) -> list[str] | None:
    if not root_folder:
        st.error("请先填写年报总文件夹。")
        return None
    file_names = parse_file_names(file_names_text)
    if run_mode == "单文件" and len(file_names) != 1:
        st.error("单文件模式下，请只输入一个文件名。")
        return None
    if not file_names:
        st.error("请输入至少一个年报文件名。")
        return None
    resolved_files, missing_files = resolve_file_names(root_folder, file_names)
    if missing_files:
        st.error("以下文件未在总文件夹中找到，请检查文件名是否完整准确：")
        st.code("\n".join(missing_files), language="text")
        return None
    return resolved_files


def render_run_summary() -> None:
    result = st.session_state.get("run_result")
    if not result:
        st.info("先预览候选句，再决定是否调用 AI。")
        return

    summary = result.get("summary", {})
    stage = result.get("stage", "preview")
    st.write("当前阶段：`候选句预览`" if stage == "preview" else "当前阶段：`AI 已完成标注`")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("文档数", summary.get("document_count", 0))
    col2.metric("原始召回句数", summary.get("raw_recall_total", 0))
    col3.metric("最终候选句数", summary.get("candidate_count", 0))
    col4.metric("异常文件", summary.get("error_count", 0))

    st.write(
        f"本轮最少目标句数：`{summary.get('min_total_sentences', 0)}`，"
        f"允许上限：`{summary.get('max_total_allowed', 0)}`，"
        f"动态每篇基础配额：`{summary.get('dynamic_target_per_file', 0)}`。"
    )

    if stage == "preview":
        st.info("先看每篇年报选了多少句；如果你觉得模式和数量都可以，再点左侧“确认后开始 AI 标注”。")

    if summary.get("files_below_target"):
        st.subheader("低于每篇期望句子数的文件")
        st.dataframe(pd.DataFrame(summary["files_below_target"]), use_container_width=True)

    if summary.get("files_above_target"):
        st.subheader("高于每篇期望句子数的文件")
        st.dataframe(pd.DataFrame(summary["files_above_target"]), use_container_width=True)

    if summary.get("documents_without_candidates"):
        st.subheader("没有进入最终候选池的文件")
        st.code("\n".join(summary["documents_without_candidates"]), language="text")

    if result["errors"]:
        st.subheader("异常文件")
        st.dataframe(pd.DataFrame([error.__dict__ for error in result["errors"]]), use_container_width=True)


def render_overview() -> None:
    result = st.session_state.get("run_result")
    if not result:
        st.info("暂无结果。")
        return

    records = get_active_records()
    if not records:
        st.warning("当前没有可展示的候选句。")
        return

    stage = result.get("stage", "preview")
    if stage == "preview":
        df = build_preview_dataframe(records)
    else:
        df = build_analysis_dataframe(records)

    summary = result.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("文档数", summary.get("document_count", 0))
    col2.metric("当前候选句数", len(df))
    col3.metric("已排除文件", len(st.session_state.get("excluded_files", [])))
    col4.metric("状态", "预览" if stage == "preview" else "已标注")

    st.subheader("按文件统计")
    file_stats = build_file_stats_dataframe(result, df)
    target = summary.get("target_sentences_per_file", 15)
    file_stats["状态"] = file_stats["最终入选数"].apply(lambda x: classify_file_status(x, target))

    st.dataframe(file_stats, use_container_width=True, hide_index=True)
    st.altair_chart(build_file_usage_chart(file_stats, target), use_container_width=True)
    st.caption("先确认每篇年报的原始召回数、最终入选数、编码和告警，再决定是否调用 AI。")

    selected_files = file_stats["显示名称"].tolist()
    excluded = st.multiselect(
        "排除不想继续审核/导出的年报",
        options=selected_files,
        default=[f for f in st.session_state.get("excluded_files", []) if f in selected_files],
    )
    if excluded != st.session_state.get("excluded_files", []):
        st.session_state.excluded_files = excluded
        st.rerun()

    visible_stats = file_stats[~file_stats["显示名称"].isin(st.session_state.get("excluded_files", []))]
    if visible_stats.empty:
        st.info("当前所有文件都已被排除。")
        return
    selected_file = st.selectbox("选择一篇年报查看候选句明细", visible_stats["显示名称"].tolist(), index=0)
    if stage == "preview":
        detail_df = df[df["显示名称"] == selected_file][
            ["句子内容", "matched_keywords", "rule_flags", "rule_label", "rule_reason"]
        ].copy()
    else:
        detail_df = df[df["显示名称"] == selected_file][
            ["句子内容", "final_label", "reviewed_label", "confidence", "判断理由", "primary_keywords", "secondary_keywords"]
        ].copy()
    st.dataframe(detail_df, use_container_width=True)


def render_review() -> None:
    result = st.session_state.get("run_result")
    if not result or result.get("stage") != "judged":
        st.info("请先完成候选句预览，并确认后调用 AI 标注。")
        return

    df = get_active_analysis_df()
    if df is None or df.empty:
        st.info("暂无可审核数据。")
        return

    source_options = ["全部"] + sorted(df["来源文件"].dropna().unique().tolist())
    label_options = ["全部", 0, 1]
    source_filter = st.selectbox("来源文件", source_options, index=0)
    label_filter = st.selectbox("当前标签", label_options, index=0)
    confidence_threshold = st.slider("最低置信度", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    keyword_query = st.text_input("关键词检索", value="")

    filtered = df.copy()
    if source_filter != "全部":
        filtered = filtered[filtered["来源文件"] == source_filter]
    if label_filter != "全部":
        filtered = filtered[effective_label_series(filtered) == label_filter]
    filtered = filtered[filtered["confidence"] >= confidence_threshold]
    if keyword_query:
        mask = filtered["matched_keywords"].fillna("").str.contains(keyword_query, case=False)
        mask = mask | filtered["primary_keywords"].fillna("").str.contains(keyword_query, case=False)
        mask = mask | filtered["secondary_keywords"].fillna("").str.contains(keyword_query, case=False)
        mask = mask | filtered["句子内容"].fillna("").str.contains(keyword_query, case=False)
        filtered = filtered[mask]

    edited = st.data_editor(
        filtered[
            [
                "record_id",
                "来源文件",
                "id",
                "year",
                "句子内容",
                "sentence_index",
                "char_position",
                "primary_keywords",
                "secondary_keywords",
                "final_label",
                "reviewed_label",
                "confidence",
                "判断理由",
                "context_before",
                "context_after",
                "review_note",
            ]
        ],
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "reviewed_label": st.column_config.SelectboxColumn("人工复核标签", options=[None, 0, 1]),
            "review_note": st.column_config.TextColumn("审核备注"),
            "句子内容": st.column_config.TextColumn("句子内容", width="large"),
            "primary_keywords": st.column_config.TextColumn("一类关键词命中", width="medium"),
            "secondary_keywords": st.column_config.TextColumn("二类关键词命中", width="medium"),
            "context_before": st.column_config.TextColumn("上文", width="large"),
            "context_after": st.column_config.TextColumn("下文", width="large"),
        },
        hide_index=True,
        disabled=[
            "record_id",
            "来源文件",
            "id",
            "year",
            "句子内容",
            "sentence_index",
            "char_position",
            "primary_keywords",
            "secondary_keywords",
            "final_label",
            "confidence",
            "判断理由",
            "context_before",
            "context_after",
        ],
        key="review_editor",
    )
    if st.button("保存当前审核修改"):
        base_df = st.session_state.get("edited_analysis_df")
        merged = base_df.set_index("record_id")
        edited = edited.set_index("record_id")
        for record_id, row in edited.iterrows():
            merged.loc[record_id, "reviewed_label"] = row["reviewed_label"]
            merged.loc[record_id, "review_note"] = row["review_note"]
        st.session_state.edited_analysis_df = merged.reset_index()
        st.session_state.export_payload = None
        st.session_state.export_record_ids = []
        st.success("审核修改已保存到当前会话。")


def render_export(output_path: str) -> None:
    result = st.session_state.get("run_result")
    if not result:
        st.info("请先完成候选句预览或 AI 标注后再导出。")
        return

    if result.get("stage") == "preview":
        records = get_active_records()
        if not records:
            st.info("当前没有可导出的候选句。")
            return
        payload = export_preview_xlsx(records)
        st.download_button(
            "下载候选句预览表",
            data=payload,
            file_name="annual_report_preview_candidates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            on_click="ignore",
            key="download_preview_xlsx",
        )
        st.dataframe(build_preview_export_dataframe(records), use_container_width=True, hide_index=True)
        if st.button("写入候选句预览到导出路径"):
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            export_preview_xlsx(records, str(target))
            st.success(f"已写出候选句预览到: {target}")
        return

    edited_df = get_active_analysis_df()
    if edited_df is None or edited_df.empty:
        st.info("请先完成 AI 标注后再导出。")
        return

    records = sync_records_with_edits(result["judgments"], edited_df)
    record_ids = [record.record_id for record in records]
    if st.session_state.get("export_payload") is None or st.session_state.get("export_record_ids") != record_ids:
        st.session_state.export_payload = export_dual_xlsx(records)
        st.session_state.export_record_ids = record_ids
    payload = st.session_state.export_payload

    st.download_button(
        "下载双表 Excel",
        data=payload,
        file_name="annual_report_labeling_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        on_click="ignore",
        key="download_export_xlsx",
    )

    if st.button("写入到导出路径"):
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        export_dual_xlsx(records, str(target))
        st.success(f"已写出到: {target}")


def parse_file_names(file_names_text: str) -> list[str]:
    return [line.strip() for line in file_names_text.splitlines() if line.strip()]


def resolve_file_names(root_folder: str, file_names: list[str]) -> tuple[list[str], list[str]]:
    root = Path(root_folder)
    resolved: list[str] = []
    missing: list[str] = []
    for file_name in file_names:
        candidate = root / file_name
        if candidate.exists():
            resolved.append(str(candidate.resolve()))
        else:
            missing.append(file_name)
    return resolved, missing


def get_active_records() -> list:
    result = st.session_state.get("run_result")
    if not result:
        return []
    excluded = set(st.session_state.get("excluded_files", []))
    records = result["judgments"] if result.get("stage") == "judged" else result["sentences"]
    if not excluded:
        return records
    return [record for record in records if file_name_only(record.source_file) not in excluded]


def build_preview_dataframe(records: list) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "record_id": record.record_id,
                "来源文件": file_name_only(record.source_file),
                "显示名称": build_display_name(record.source_file, record.year, record.company_name),
                "句子内容": record.sentence,
                "matched_keywords": ", ".join(record.matched_keywords),
                "rule_flags": ", ".join(record.rule_flags),
                "rule_label": record.rule_label,
                "rule_reason": record.rule_reason,
            }
        )
    return pd.DataFrame(rows)


def get_active_analysis_df() -> pd.DataFrame | None:
    df = st.session_state.get("edited_analysis_df")
    if df is None:
        return None
    excluded = set(st.session_state.get("excluded_files", []))
    if not excluded:
        return df
    return df[~df["来源文件"].isin(excluded)].copy()


def sync_records_with_edits(records: list[JudgmentRecord], edited_df: pd.DataFrame) -> list[JudgmentRecord]:
    review_map = edited_df.set_index("record_id")[["reviewed_label", "review_note"]].to_dict("index")
    allowed_ids = set(edited_df["record_id"].tolist())
    synced: list[JudgmentRecord] = []
    for record in records:
        if record.record_id not in allowed_ids:
            continue
        review = review_map.get(record.record_id, {})
        record.reviewed_label = normalize_review_value(review.get("reviewed_label"))
        record.review_note = str(review.get("review_note") or "")
        synced.append(record)
    return synced


def normalize_review_value(value):
    if value in ("", None) or pd.isna(value):
        return None
    if value in (0, 1):
        return int(value)
    if str(value) in {"0", "1"}:
        return int(value)
    return None


def effective_label_series(df: pd.DataFrame) -> pd.Series:
    reviewed = df["reviewed_label"]
    return reviewed.where(reviewed.notna(), df["final_label"])


def classify_file_status(count: int, target: int) -> str:
    if count < target:
        return "过少"
    if count > target:
        return "过多"
    return "合适"


def build_file_usage_chart(file_stats: pd.DataFrame, target: int):
    chart_data = file_stats.copy()
    chart_data["目标句数"] = target
    color_scale = alt.Scale(
        domain=["过少", "合适", "过多"],
        range=["#d73027", "#1a9850", "#e6ab02"],
    )
    bars = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            y=alt.Y("显示名称:N", sort="-x", title="年份 + 企业", axis=alt.Axis(labelLimit=280)),
            x=alt.X("最终入选数:Q", title="最终选用句子数"),
            color=alt.Color("状态:N", scale=color_scale, legend=alt.Legend(title="状态")),
            tooltip=["显示名称", "来源文件", "原始召回数", "最终入选数", "编码", "解码告警", "状态"],
        )
    )
    target_line = alt.Chart(chart_data).mark_rule(strokeDash=[6, 4], color="#4c78a8").encode(x="目标句数:Q")
    return (bars + target_line).properties(height=320)


def file_name_only(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]


def build_file_stats_dataframe(result: dict, df: pd.DataFrame) -> pd.DataFrame:
    summary = result.get("summary", {})
    file_stats = pd.DataFrame(summary.get("file_stats", []))
    if file_stats.empty:
        file_stats = (
            df.groupby("来源文件")
            .agg(最终入选数=("句子内容", "count"))
            .reset_index()
        )
        file_stats["原始召回数"] = file_stats["最终入选数"]
        file_stats["编码"] = None
        file_stats["解码告警"] = ""
        file_stats["显示名称"] = file_stats["来源文件"].apply(lambda value: build_display_name(value))
    else:
        file_stats = file_stats.rename(
            columns={
                "source_file": "完整路径",
                "raw_recall_count": "原始召回数",
                "selected_count": "最终入选数",
                "target": "目标句数",
                "encoding": "编码",
                "warnings": "解码告警",
            }
        )
        file_stats["来源文件"] = file_stats["完整路径"].apply(file_name_only)
        document_meta = {
            document.source_file: {
                "year": document.year,
                "company_name": document.company_name,
            }
            for document in result.get("documents", [])
        }
        file_stats["显示名称"] = file_stats["完整路径"].apply(
            lambda path: build_display_name(
                path,
                document_meta.get(path, {}).get("year"),
                document_meta.get(path, {}).get("company_name"),
            )
        )
    if "confidence" in df.columns:
        confidence_df = df.groupby("显示名称", as_index=False)["confidence"].mean().rename(columns={"confidence": "平均置信度"})
        file_stats = file_stats.merge(confidence_df, on="显示名称", how="left")
    else:
        file_stats["平均置信度"] = None
    preferred_columns = ["显示名称", "来源文件", "原始召回数", "最终入选数", "目标句数", "编码", "解码告警", "平均置信度"]
    for column in preferred_columns:
        if column not in file_stats.columns:
            file_stats[column] = None
    return file_stats[preferred_columns].sort_values(["最终入选数", "原始召回数"], ascending=False)


def build_display_name(path: str, year: str | None = None, company_name: str | None = None) -> str:
    if not year or not company_name:
        parsed = parse_filename(path)
        year = year or parsed.get("year")
        company_name = company_name or parsed.get("company_name")
    if year and company_name:
        return f"{year} {company_name}"
    return file_name_only(path)
