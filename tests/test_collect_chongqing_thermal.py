import unittest

from tools.collect_chongqing_thermal import discover_thermal_attachments


class ChongqingThermalCollectorTests(unittest.TestCase):
    def test_discovers_pdf_attachment_links_embedded_in_script_strings(self):
        html = """
        <script>
        var hasFJ='<a href="./P020230703659249570817.docx">附件1：技术细则.docx</a><BR/>
        <a href="./P020230703659249750923.pdf">附件2：《重庆市建筑门窗幕墙热工参数目录（2023版）》.pdf</a>';
        </script>
        """

        attachments = discover_thermal_attachments(
            html,
            page_url="https://zfcxjw.cq.gov.cn/zwxx_166/gsgg/202307/t20230703_12117419.html",
        )

        self.assertEqual(len(attachments), 1)
        self.assertEqual(
            attachments[0].url,
            "https://zfcxjw.cq.gov.cn/zwxx_166/gsgg/202307/P020230703659249750923.pdf",
        )
        self.assertIn("2023版", attachments[0].title)


if __name__ == "__main__":
    unittest.main()
