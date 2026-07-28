(function registerExportResult(root, factory) {
  "use strict";

  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.AnnotationExportResult = api;
  }
})(
  typeof globalThis === "object" ? globalThis : this,
  () => {
    "use strict";

    function formatExportDetail(result) {
      return (
        `${result.sample_count} 个样本 · ${result.experiment_type}\n` +
        `导出路径：${result.export_path}`
      );
    }

    return Object.freeze({ formatExportDetail });
  }
);
