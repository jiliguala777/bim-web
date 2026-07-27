import csv
import tempfile
import unittest
from pathlib import Path

from tools.thermal_candidate_extractor import (
    ThermalCandidate,
    collect_candidates_from_manifest,
    collect_candidates_from_file,
    ssl_context_for_verify,
    detect_html_encoding,
    extract_chongqing_window_catalog_candidates,
    extract_candidates_from_html,
    extract_candidates_from_text,
    write_candidates_csv,
    normalize_unit,
)


class ThermalCandidateExtractorTests(unittest.TestCase):
    def test_extracts_u_value_and_lambda_from_publicity_html_table(self):
        html = """
        <table>
          <tr>
            <td>序号</td><td>企业名称</td><td>材料(产品)名称</td>
            <td>适用范围</td><td>热传递性能</td><td>报告编号</td>
          </tr>
          <tr>
            <td>1</td><td>苏州立邦新材料科技有限公司</td>
            <td>立邦节能装饰一体板(岩棉)系统</td><td>建筑外墙外保温</td>
            <td>1.01W/(㎡·K)</td><td>NBEC-2024CX-0791-1</td>
          </tr>
          <tr>
            <td>2</td><td>湖南弘阁节能科技有限公司</td>
            <td>Ⅰ型水性节能环保装饰一体涂料</td><td>建筑墙体保温</td>
            <td>0.048W/(m·K)</td><td>ZJTA5310217730223GR</td>
          </tr>
        </table>
        """

        rows = extract_candidates_from_html(
            html,
            source_url="https://xtjs.xiangtan.gov.cn/example.html",
            source_title="湘潭市建筑节能材料公示",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].component_category, "wall_insulation")
        self.assertEqual(rows[0].parameter_type, "thermal_transmittance_u")
        self.assertEqual(rows[0].name, "立邦节能装饰一体板(岩棉)系统")
        self.assertEqual(rows[0].value, 1.01)
        self.assertEqual(rows[0].unit, "W/(m2·K)")
        self.assertEqual(rows[0].source_report_id, "NBEC-2024CX-0791-1")

        self.assertEqual(rows[1].parameter_type, "thermal_conductivity_lambda")
        self.assertEqual(rows[1].value, 0.048)
        self.assertEqual(rows[1].unit, "W/(m·K)")

    def test_extracts_values_when_legacy_page_corrupts_square_meter_symbols(self):
        html = """
        <table>
          <tr><td>col0</td><td>col1</td><td>col2</td><td>product</td><td>std</td><td>agency</td><td>report</td><td>scope</td><td>addr</td><td>term</td><td>thermal</td></tr>
          <tr><td>1</td><td>XTJNGS</td><td>maker</td><td>wall system</td><td>JG/T287</td><td>agency</td><td>NBEC-1</td><td>external wall insulation</td><td>addr</td><td>date</td><td>1.01W/(�O��K)</td></tr>
          <tr><td>2</td><td>XTJNGS</td><td>maker</td><td>coating</td><td>DBJ</td><td>agency</td><td>RPT-2</td><td>wall insulation</td><td>addr</td><td>date</td><td>0.048W/(m��K)</td></tr>
        </table>
        """

        rows = extract_candidates_from_html(html, source_url="https://example.gov.cn", source_title="legacy")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].parameter_type, "thermal_transmittance_u")
        self.assertEqual(rows[0].unit, "W/(m2·K)")
        self.assertEqual(rows[0].source_report_id, "NBEC-1")
        self.assertEqual(rows[1].parameter_type, "thermal_conductivity_lambda")
        self.assertEqual(rows[1].unit, "W/(m·K)")

    def test_extracts_roof_floor_door_and_window_values_from_text(self):
        text = """
        屋面传热系数满足要求 K = 0.29 W/(m2·K)。
        周边地面热阻 R = 1.29 (m2·K)/W。
        非透光外门类型 多功能户门 传热系数 1.50 W/(m2·K)。
        外窗（含透光幕墙）太阳得热系数 Shgc = 0.44。
        """

        rows = extract_candidates_from_text(
            text,
            source_url="https://example.gov.cn/report.pdf",
            source_title="公共建筑节能计算分析报告书",
        )

        pairs = {(row.component_category, row.parameter_type): row for row in rows}
        self.assertEqual(pairs[("roof_u", "thermal_transmittance_u")].value, 0.29)
        self.assertEqual(pairs[("floor_contact_type", "thermal_resistance_r")].value, 1.29)
        self.assertEqual(pairs[("door_u", "thermal_transmittance_u")].value, 1.5)
        self.assertEqual(pairs[("window_shgc", "solar_heat_gain_shgc")].value, 0.44)

    def test_writes_review_csv_with_stable_columns(self):
        candidate = ThermalCandidate(
            component_category="roof_u",
            parameter_type="thermal_transmittance_u",
            name="屋面传热系数",
            value=0.29,
            unit="W/(m2·K)",
            source_title="报告",
            source_url="https://example.gov.cn/report.pdf",
            source_doc="report.pdf",
            source_report_id="",
            source_text="屋面传热系数 K = 0.29 W/(m2·K)",
            notes="",
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "thermal_parameter_candidates.csv"
            write_candidates_csv([candidate], output)
            with output.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["component_category"], "roof_u")
        self.assertEqual(rows[0]["parameter_type"], "thermal_transmittance_u")
        self.assertEqual(rows[0]["review_status"], "pending")

    def test_collects_candidates_from_html_file_by_suffix(self):
        html = """
        <table>
          <tr><td>材料(产品)名称</td><td>热传递性能</td><td>报告编号</td></tr>
          <tr><td>保温屋面构造</td><td>0.29W/(㎡·K)</td><td>RPT-1</td></tr>
        </table>
        """

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "source.html"
            input_path.write_text(html, encoding="utf-8")
            rows = collect_candidates_from_file(
                input_path,
                source_url="file://source.html",
                source_title="本地样本",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].component_category, "roof_u")
        self.assertEqual(rows[0].parameter_type, "thermal_transmittance_u")

    def test_collects_candidates_from_manifest_with_local_sources(self):
        html = """
        <table>
          <tr><td>材料(产品)名称</td><td>热传递性能</td><td>报告编号</td></tr>
          <tr><td>保温外门</td><td>1.20W/(㎡·K)</td><td>RPT-DOOR</td></tr>
        </table>
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.html"
            manifest_path = root / "sources.csv"
            source_path.write_text(html, encoding="utf-8")
            manifest_path.write_text(
                "title,url,local_path,verify_ssl\n"
                f"本地历史样本,https://example.gov.cn/source.html,{source_path},true\n",
                encoding="utf-8",
            )

            rows = collect_candidates_from_manifest(manifest_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source_title, "本地历史样本")
        self.assertEqual(rows[0].source_url, "https://example.gov.cn/source.html")

    def test_can_build_insecure_ssl_context_for_certificate_chain_exceptions(self):
        context = ssl_context_for_verify(False)

        self.assertFalse(context.check_hostname)
        self.assertEqual(context.verify_mode, 0)

    def test_detects_meta_charset_when_http_header_omits_charset(self):
        raw = b'<html><head><meta charset="gb2312"></head><body></body></html>'

        self.assertEqual(detect_html_encoding(raw, "text/html"), "gb18030")

    def test_normalizes_units_with_corrupt_middle_dot(self):
        self.assertEqual(normalize_unit("W/(m2��K)"), "W/(m2·K)")
        self.assertEqual(normalize_unit("W/(m��K)"), "W/(m·K)")

    def test_extracts_chongqing_window_catalog_ug_and_window_u_levels(self):
        text = """
        典型门窗幕墙热工参数目录
        序号 玻璃类型 玻璃传热系数 U g[W/（m2·K）]
        典型门窗幕墙传热系数U f [W/（m2·K）]
        1.0级 2.0级 3.0级 4.0级 5.0级
        1 两玻一腔 中空玻璃 普通 6透明+12A+6透明 2.59 2.4 2.7 2.9 3.2 3.4
        6单银Low-E+12Ar+6透明（全自动化封装暖边条） 1.44 1.6 1.8 2.1 2.3 2.6
        18 单银 6单银Low-E+12A+6透明 1.72 1.8 2.0 2.3 2.5 2.8
        """

        rows = extract_chongqing_window_catalog_candidates(
            text,
            source_url="https://zfcxjw.cq.gov.cn/example.pdf",
            source_title="重庆市建筑门窗幕墙热工参数目录（2023版）",
            source_doc="cq.pdf",
        )

        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[0].component_category, "window_u")
        self.assertEqual(rows[0].parameter_type, "thermal_transmittance_u")
        self.assertIn("玻璃Ug", rows[0].name)
        self.assertEqual(rows[0].value, 2.59)
        self.assertEqual(rows[1].name, "6透明+12A+6透明 - 典型门窗幕墙1.0级")
        self.assertEqual(rows[1].value, 2.4)

    def test_extracts_chongqing_glass_optical_shgc_and_ug(self):
        text = """
        典型玻璃的光学、热工性能参数表
        序号 类型 可见光透射比τv 太阳得热系数SHGC 传热系数Ug [W/（m2•K）]
        1 透明 6透明玻璃 0.90 0.85 — 5.15 — 0.15 0.12
        39 单银 6高透光单银Low-E+12A+6透明 0.68 0.46 — 1.72 0.030 — —
        """

        rows = extract_chongqing_window_catalog_candidates(
            text,
            source_url="https://zfcxjw.cq.gov.cn/example.pdf",
            source_title="重庆市建筑门窗幕墙热工参数目录（2023版）",
            source_doc="cq.pdf",
        )

        pairs = {(row.name, row.parameter_type): row for row in rows}
        self.assertEqual(pairs[("6透明玻璃 - 玻璃SHGC", "solar_heat_gain_shgc")].value, 0.85)
        self.assertEqual(pairs[("6透明玻璃 - 玻璃Ug", "thermal_transmittance_u")].value, 5.15)
        self.assertEqual(
            pairs[("6高透光单银Low-E+12A+6透明 - 玻璃SHGC", "solar_heat_gain_shgc")].value,
            0.46,
        )


if __name__ == "__main__":
    unittest.main()
