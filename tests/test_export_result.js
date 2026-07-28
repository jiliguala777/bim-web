"use strict";

const assert = require("node:assert/strict");
const {
  formatExportDetail,
} = require("../annotation_tool/static/export_result.js");

const exportPath =
  "G:\\bim网页\\标注数据\\exports\\project-bce78876-20260728T065351Z";

assert.equal(
  formatExportDetail({
    sample_count: 1,
    experiment_type: "single_page_overfit",
    export_path: exportPath,
  }),
  `1 个样本 · single_page_overfit\n导出路径：${exportPath}`,
);
