from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from report_labeler.export import build_analysis_dataframe, build_preview_export_dataframe, build_submission_dataframe
from report_labeler.io_utils import detect_and_read_text, parse_filename
from report_labeler.models import ModelConfig, PipelineConfig
from report_labeler.pipeline import judge_preview, preview_files, run_single
from report_labeler.preprocess import clean_text, is_fragment_like, normalize_sentence_text, split_sentences
from report_labeler.rules import evaluate_rules
from report_labeler.ui import parse_file_names, resolve_file_names


class PipelineTests(unittest.TestCase):
    def test_parse_filename(self):
        data = parse_filename("000001_2024_平安银行_2024年年度报告_2025-03-15.txt")
        self.assertEqual(data["stock_id"], "000001")
        self.assertEqual(data["year"], "2024")
        self.assertEqual(data["company_name"], "平安银行")

    def test_detect_utf16(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.txt"
            path.write_text("测试内容", encoding="utf-16")
            text, encoding, warnings = detect_and_read_text(str(path), ["utf-16", "utf-8"])
            self.assertEqual(text, "测试内容")
            self.assertEqual(encoding, "utf-16")
            self.assertEqual(warnings, [])

    def test_detect_utf8_without_bom_is_not_misread_as_utf16(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo_utf8.txt"
            path.write_text("2019 年年度报告\n公司利用数字化管理手段提升生产效率。", encoding="utf-8")
            text, encoding, warnings = detect_and_read_text(
                str(path),
                ["utf-16", "utf-8-sig", "utf-8", "gb18030", "gbk"],
            )
            self.assertEqual(encoding, "utf-8")
            self.assertIn("数字化管理", text)

    def test_clean_and_split(self):
        text = "1\n目录\n公司采用云平台实现生产优化。\n未来将继续推进数字化。"
        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)
        self.assertTrue(any("公司采用云平台实现生产优化。" in s for s in sentences))
        self.assertTrue(any("未来将继续推进数字化。" in s for s in sentences))

    def test_normalize_sentence_text_removes_inner_chinese_spaces(self):
        text = "利用 PDM、CAE、NVH 等数字 化 管理 手段，提升在 线 检 测能力。"
        normalized = normalize_sentence_text(text)
        self.assertIn("数字化管理手段", normalized)
        self.assertIn("在线检测能力", normalized)

    def test_normalize_sentence_text_strips_fragment_prefix_noise(self):
        self.assertEqual(normalize_sentence_text("模式公司利用平台建设提升效率。"), "公司利用平台建设提升效率。")
        self.assertEqual(normalize_sentence_text("③持续推进智能制造。"), "持续推进智能制造。")

    def test_preview_keeps_full_sentence_without_truncation(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_测试公司_2024年年度报告_2025-01-01.txt"
            long_sentence = (
                "公司采用工业互联网平台对生产设备进行实时监测，并通过数据分析、在线检测、工艺优化、"
                "设备状态监测、预测性维护和质量管控等多种方式持续提升生产效率与产品质量，"
                "同时在多个车间协同推进智能制造能力建设。"
            )
            path.write_text(long_sentence, encoding="utf-16")
            preview = preview_files(
                [str(path)],
                PipelineConfig(sentence_max_chars=50, min_total_sentences=1, target_sentences_per_file=1),
            )
            self.assertEqual(preview["sentences"][0].sentence, long_sentence)

    def test_split_sentences_keeps_cross_line_long_sentence_together(self):
        text = (
            "公司采用工业互联网平台对生产设备进行实时监测并持续优化工艺流程\n"
            "同时通过在线检测和数据分析提升生产效率与产品质量。"
        )
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 1)
        self.assertIn("持续优化工艺流程同时通过在线检测", sentences[0])

    def test_split_sentences_merges_fragmented_tail_sentence(self):
        text = (
            "本集团采用利率互换合同以降低以浮动利率计息的融资租赁\n"
            "款的浮动利率转换成固定利率。"
        )
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 1)
        self.assertIn("融资租赁款的浮动利率转换成固定利率。", sentences[0])

    def test_is_fragment_like_identifies_obvious_broken_sentences(self):
        self.assertTrue(is_fragment_like("款的浮动利率转换成固定利率"))
        self.assertTrue(is_fragment_like("自筹资金锦江之星 BI 商务智能平台项目工程及其他"))
        self.assertFalse(is_fragment_like("公司采用工业互联网平台对生产设备进行实时监测。"))

    def test_rule_positive(self):
        result = evaluate_rules("公司采用工业互联网平台对产线设备进行实时监测并提升良品率。")
        self.assertEqual(result.label, 1)

    def test_rule_negative_future(self):
        result = evaluate_rules("公司将持续推进工业互联网平台建设，探索未来发展方向。")
        self.assertEqual(result.label, 0)

    def test_finance_sentence_is_excluded_from_candidates(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_测试公司_2024年年度报告_2025-01-01.txt"
            path.write_text(
                "融资租赁租入的固定资产，按租赁开始日租赁资产公允价值与最低租赁付款额的现值两者中较低者，作为入账价值。\n"
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            sentences = [item.sentence for item in preview["sentences"]]
            self.assertTrue(any("工业互联网平台" in sentence for sentence in sentences))
            self.assertFalse(any("融资租赁" in sentence for sentence in sentences))

    def test_secondary_keywords_expand_recall(self):
        result = evaluate_rules("公司持续推进数字底座建设，强化数据平台与智能平台能力。")
        self.assertTrue("secondary_keyword_hit" in result.flags)
        self.assertGreaterEqual(len(result.matched_keywords), 1)

    def test_ascii_keyword_boundary_avoids_false_positive_5g(self):
        result = evaluate_rules('"Metropolo" chain Hotel opened 62, "Jinjiang Inn" chain Hotel opened 1,075, "Goldmet')
        self.assertNotIn("5G", result.matched_keywords)

    def test_weak_secondary_only_sentence_is_not_selected(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_testco_2024_report_2025-01-01.txt"
            path.write_text(
                "利用估值专家评估确认上述公允价值的评估方法和模型。\n"
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            sentences = [item.sentence for item in preview["sentences"]]
            self.assertTrue(any("工业互联网平台" in sentence for sentence in sentences))
            self.assertFalse(any("评估方法和模型" in sentence for sentence in sentences))

    def test_finance_heavy_primary_keyword_sentence_is_not_selected(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_testco_2024_report_2025-01-01.txt"
            path.write_text(
                "实质上转移了与资产所有权有关的全部风险和报酬的租赁为融资租赁。\n"
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            sentences = [item.sentence for item in preview["sentences"]]
            self.assertTrue(any("工业互联网平台" in sentence for sentence in sentences))
            self.assertFalse(any("融资租赁" in sentence for sentence in sentences))

    def test_policy_sentence_is_not_selected(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_testco_2024_report_2025-01-01.txt"
            path.write_text(
                "国务院印发了智能制造发展规划，提出推进数字化转型和网络化协同。\n"
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            sentences = [item.sentence for item in preview["sentences"]]
            self.assertTrue(any("工业互联网平台" in sentence for sentence in sentences))
            self.assertFalse(any("国务院印发" in sentence for sentence in sentences))

    def test_disclosure_sentence_is_not_selected(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_testco_2024_report_2025-01-01.txt"
            path.write_text(
                "对于授予的不存在活跃市场的期权等权益工具，采用期权定价模型等确定其公允价值。\n"
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            sentences = [item.sentence for item in preview["sentences"]]
            self.assertTrue(any("工业互联网平台" in sentence for sentence in sentences))
            self.assertFalse(any("期权定价模型" in sentence for sentence in sentences))

    def test_external_service_sentence_is_not_selected(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_testco_2024_report_2025-01-01.txt"
            path.write_text(
                "公司通过快速响应客户需求和提供优质持续的服务，提高客户满意度。\n"
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            sentences = [item.sentence for item in preview["sentences"]]
            self.assertTrue(any("工业互联网平台" in sentence for sentence in sentences))
            self.assertFalse(any("客户满意度" in sentence for sentence in sentences))

    def test_capital_investment_sentence_is_not_selected(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_testco_2024_report_2025-01-01.txt"
            path.write_text(
                "本年度，公司投资1000万元参与了某产业基金的B+轮融资。\n"
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            sentences = [item.sentence for item in preview["sentences"]]
            self.assertTrue(any("工业互联网平台" in sentence for sentence in sentences))
            self.assertFalse(any("产业基金" in sentence for sentence in sentences))

    def test_primary_keyword_mode_excludes_secondary_only_sentences(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_测试公司_2024年年度报告_2025-01-01.txt"
            path.write_text(
                "公司持续推进数字底座建设，强化数据平台与智能平台能力。\n"
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。",
                encoding="utf-16",
            )
            preview = preview_files(
                [str(path)],
                PipelineConfig(filter_strength="一类关键词", min_total_sentences=5, target_sentences_per_file=5),
            )
            self.assertTrue(all("secondary_keyword_hit" not in item.rule_flags or "primary_keyword_hit" in item.rule_flags for item in preview["sentences"]))
            self.assertTrue(any("工业互联网" in item.sentence for item in preview["sentences"]))

    def test_primary_keywords_in_realistic_sentences_are_recalled(self):
        result = evaluate_rules(
            "长春-苏州人才联合培养建立长效机制，利用PDM、CAE、NVH等数字 化 管理手段，"
            "建立数字化科研开发体系。"
        )
        self.assertIn("数字化管理", result.matched_keywords)
        self.assertIn("primary_keyword_hit", result.flags)

        result2 = evaluate_rules("推广MOD工具应用，提升生产效率")
        self.assertIn("生产效率", result2.matched_keywords)

    def test_title_like_lines_are_filtered_but_content_is_kept(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_测试公司_2024年年度报告_2025-01-01.txt"
            path.write_text(
                "数字化管理\n"
                "公司利用数字化管理手段，提升生产效率。\n",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            self.assertEqual(len(preview["sentences"]), 1)
            self.assertIn("提升生产效率", preview["sentences"][0].sentence)

    def test_numbered_heading_prefix_is_stripped_from_body_sentence(self):
        text = (
            "3、人才联合培养机制成效显著\n"
            "长春-苏州人才联合培养建立长效机制，在探索技术创新的道路上不断深化研发体系改革，"
            "积极探索校企合作模式，利用 PDM、CAE、NVH 等数字化管理手段，建立数字化科研开发体系。"
        )
        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)
        self.assertEqual(len(sentences), 1)
        self.assertNotIn("人才联合培养机制成效显著", sentences[0])
        result = evaluate_rules(sentences[0])
        self.assertIn("数字化管理", result.matched_keywords)

    def test_two_stage_preview_then_judge(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_测试公司_2024年年度报告_2025-01-01.txt"
            path.write_text(
                "公司采用工业互联网平台对生产设备进行实时监测，提升生产效率。\n"
                "公司将继续推进云平台建设。",
                encoding="utf-16",
            )
            preview = preview_files([str(path)], PipelineConfig())
            self.assertEqual(preview["stage"], "preview")
            self.assertGreaterEqual(len(preview["sentences"]), 2)

            judged = judge_preview(preview, ModelConfig(provider="mock"))
            self.assertEqual(judged["stage"], "judged")
            self.assertEqual(len(judged["judgments"]), len(preview["sentences"]))

            submission_df = build_submission_dataframe(judged["judgments"])
            analysis_df = build_analysis_dataframe(judged["judgments"])
            preview_df = build_preview_export_dataframe(preview["sentences"])
            self.assertEqual(
                list(submission_df.columns),
                ["句子内容", "来源文件", "人工标注标签", "id", "year", "判断理由"],
            )
            self.assertIn("显示名称", analysis_df.columns)
            self.assertIn("primary_keywords", analysis_df.columns)
            self.assertIn("secondary_keywords", analysis_df.columns)
            self.assertIn("判断理由", analysis_df.columns)
            self.assertIn("一类关键词命中", preview_df.columns)
            self.assertIn("二类关键词命中", preview_df.columns)

    def test_run_single_keeps_backward_compatibility(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001_2024_测试公司_2024年年度报告_2025-01-01.txt"
            path.write_text("公司采用工业互联网平台对生产设备进行实时监测。", encoding="utf-16")
            result = run_single(str(path), ModelConfig(provider="mock"), PipelineConfig())
            self.assertEqual(result["stage"], "judged")
            self.assertGreaterEqual(len(result["judgments"]), 1)

    def test_file_name_input_workflow(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path1 = root / "000001_2024_测试公司A_2024年年度报告_2025-01-01.txt"
            path2 = root / "000002_2024_测试公司B_2024年年度报告_2025-01-02.txt"
            path1.write_text("公司采用工业互联网平台进行生产优化。", encoding="utf-16")
            path2.write_text("公司将推进工业互联网平台建设。", encoding="utf-16")

            file_names = parse_file_names(f"{path1.name}\n{path2.name}\n")
            resolved, missing = resolve_file_names(str(root), file_names)
            self.assertEqual(missing, [])
            self.assertEqual(len(resolved), 2)


if __name__ == "__main__":
    unittest.main()
