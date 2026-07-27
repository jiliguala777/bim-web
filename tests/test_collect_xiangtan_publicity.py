import unittest

from tools.collect_xiangtan_publicity import discover_publicity_pages


class XiangtanPublicityCollectorTests(unittest.TestCase):
    def test_discovers_only_building_energy_material_publicity_links(self):
        html = """
        <html><body>
          <a href="content_1.html">关于公布湘潭市2026年第四批建筑节能材料（产品）公示名单的通知</a>
          <a href="content_2.html">关于湘潭市绿色建筑专家库拟入库专家名单的公示</a>
          <a href="/6550/28176/content_3.html">关于公布湘潭市2025年第三批建筑节能 材料（产品）公示名单的通知</a>
          <a href="content_4.html">关于湘潭市建筑节能新材料（产品）建筑应用试点的公示</a>
        </body></html>
        """

        pages = discover_publicity_pages(
            html,
            base_url="https://xtjs.xiangtan.gov.cn/6550/28176/index.htm",
        )

        self.assertEqual(len(pages), 2)
        self.assertEqual(
            pages[0].url,
            "https://xtjs.xiangtan.gov.cn/6550/28176/content_1.html",
        )
        self.assertIn("第四批", pages[0].title)
        self.assertEqual(
            pages[1].url,
            "https://xtjs.xiangtan.gov.cn/6550/28176/content_3.html",
        )


if __name__ == "__main__":
    unittest.main()
