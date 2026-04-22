from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from report_labeler.export import build_analysis_dataframe, build_submission_dataframe
from report_labeler.io_utils import detect_and_read_text, parse_filename
from report_labeler.models import ModelConfig, PipelineConfig
from report_labeler.pipeline import judge_preview, preview_files, run_single
from report_labeler.preprocess import clean_text, split_sentences
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

    def test_rule_positive(self):
        result = evaluate_rules("公司采用工业互联网平台对产线设备进行实时监测并提升良品率。")
        self.assertEqual(result.label, 1)

    def test_rule_negative_future(self):
        result = evaluate_rules("公司将持续推进工业互联网平台建设，探索未来发展方向。")
        self.assertEqual(result.label, 0)

    def test_secondary_keywords_expand_recall(self):
        result = evaluate_rules("公司持续推进数字底座建设，强化数据平台与智能平台能力。")
        self.assertTrue("secondary_keyword_hit" in result.flags)
        self.assertGreaterEqual(len(result.matched_keywords), 1)

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
        self.assertTrue(sentences[0].startswith("长春-苏州人才联合培养建立长效机制"))
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
            self.assertEqual(
                list(submission_df.columns),
                ["句子内容", "来源文件", "人工标注标签", "id", "year", "判断理由"],
            )
            self.assertIn("显示名称", analysis_df.columns)
            self.assertIn("primary_keywords", analysis_df.columns)
            self.assertIn("secondary_keywords", analysis_df.columns)
            self.assertIn("判断理由", analysis_df.columns)

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
