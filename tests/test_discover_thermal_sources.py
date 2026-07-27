import csv
import tempfile
import unittest
from pathlib import Path

from tools.discover_thermal_sources import (
    DiscoveryResult,
    discover_from_search_html,
    discover_from_seed_html,
    normalize_result_url,
    score_source,
    write_source_manifest,
)


class DiscoverThermalSourcesTests(unittest.TestCase):
    def test_scores_high_confidence_pdf_with_window_catalog_terms(self):
        result = score_source(
            title="重庆市建筑门窗幕墙热工参数目录（2023版） PDF",
            url="https://zfcxjw.cq.gov.cn/zwxx/P020230703659249750923.pdf",
            snippet="玻璃传热系数 Ug 太阳得热系数 SHGC 典型门窗幕墙传热系数",
        )

        self.assertEqual(result.source_type, "pdf")
        self.assertEqual(result.component_hint, "window")
        self.assertIn("thermal_transmittance_u", result.parameter_hint)
        self.assertIn("solar_heat_gain_shgc", result.parameter_hint)
        self.assertGreaterEqual(result.confidence, 80)

    def test_rejects_untrusted_domains(self):
        result = score_source(
            title="建筑门窗幕墙热工参数目录",
            url="https://example.com/catalog.pdf",
            snippet="传热系数 SHGC",
        )

        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.review_status, "reject")

    def test_discovers_results_from_search_html(self):
        html = """
        <html><body>
          <a href="https://zfcxjw.cq.gov.cn/zwxx/P020230703659249750923.pdf">
            重庆市建筑门窗幕墙热工参数目录（2023版）
          </a>
          <p>玻璃传热系数 Ug，太阳得热系数 SHGC。</p>
          <a href="https://example.com/not-trusted.pdf">不可信来源</a>
        </body></html>
        """

        rows = discover_from_search_html(html)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].url, "https://zfcxjw.cq.gov.cn/zwxx/P020230703659249750923.pdf")
        self.assertEqual(rows[0].review_status, "pending")

    def test_discovers_results_from_bing_algo_blocks(self):
        html = """
        <ol id="b_results">
          <li class="b_algo">
            <h2><a href="https://zfcxjw.cq.gov.cn/zwxx/P020230703659249750923.pdf">重庆市建筑门窗幕墙热工参数目录</a></h2>
            <div class="b_caption"><p>玻璃传热系数 Ug，太阳得热系数 SHGC。</p></div>
          </li>
        </ol>
        """

        rows = discover_from_search_html(html)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].title, "重庆市建筑门窗幕墙热工参数目录")
        self.assertGreaterEqual(rows[0].confidence, 80)

    def test_discovers_pdf_links_from_trusted_seed_page(self):
        html = """
        <html><body>
          <h1>关于公开征求建筑节能设计标准意见的通知</h1>
          <a href="/zwxx/P020230703659249750923.pdf">附件2：重庆市建筑门窗幕墙热工参数目录（2023版）</a>
        </body></html>
        """

        rows = discover_from_seed_html(
            html,
            seed_url="https://zfcxjw.cq.gov.cn/zwxx/t20230703.html",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].url,
            "https://zfcxjw.cq.gov.cn/zwxx/P020230703659249750923.pdf",
        )
        self.assertEqual(rows[0].source_type, "pdf")

    def test_normalizes_bing_encoded_click_url(self):
        encoded = (
            "https://www.bing.com/ck/a?!&&u="
            "a1aHR0cHM6Ly96ZmN4ancuY3EuZ292LmNuL3p3eHgvUDAyMDIzMDcwMzY1OTI0OTc1MDkyMy5wZGY"
        )

        self.assertEqual(
            normalize_result_url(encoded),
            "https://zfcxjw.cq.gov.cn/zwxx/P020230703659249750923.pdf",
        )

    def test_writes_manifest_columns_for_review(self):
        result = DiscoveryResult(
            title="重庆目录",
            url="https://zfcxjw.cq.gov.cn/catalog.pdf",
            local_path="",
            source_type="pdf",
            component_hint="window",
            parameter_hint="thermal_transmittance_u;solar_heat_gain_shgc",
            confidence=90,
            notes="matched terms",
            review_status="pending",
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sources.csv"
            write_source_manifest([result], output)
            with output.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["title"], "重庆目录")
        self.assertEqual(rows[0]["verify_ssl"], "true")
        self.assertEqual(rows[0]["review_status"], "pending")


if __name__ == "__main__":
    unittest.main()
