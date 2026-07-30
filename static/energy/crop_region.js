(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.AnnotationCropRegion = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function canvasPointToImage(point, transform) {
    const scale = Number(transform.scale);
    if (!Number.isFinite(scale) || scale <= 0) {
      throw new Error("scale must be positive");
    }
    return {
      x: clamp(
        (Number(point.x) - Number(transform.offsetX)) / scale,
        0,
        Number(transform.width)
      ),
      y: clamp(
        (Number(point.y) - Number(transform.offsetY)) / scale,
        0,
        Number(transform.height)
      ),
    };
  }

  function normalizeCrop(start, end, width, height) {
    const x0 = clamp(Math.floor(Math.min(start.x, end.x)), 0, width);
    const y0 = clamp(Math.floor(Math.min(start.y, end.y)), 0, height);
    const x1 = clamp(Math.ceil(Math.max(start.x, end.x)), 0, width);
    const y1 = clamp(Math.ceil(Math.max(start.y, end.y)), 0, height);
    return [x0, y0, x1, y1];
  }

  function validateCrop(bbox, width, height, minSize = 128, minAreaShare = 0.01) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return false;
    const [x0, y0, x1, y1] = bbox;
    if (![x0, y0, x1, y1, width, height].every(Number.isFinite)) return false;
    if (x0 < 0 || y0 < 0 || x1 > width || y1 > height) return false;
    const cropWidth = x1 - x0;
    const cropHeight = y1 - y0;
    return (
      cropWidth >= minSize
      && cropHeight >= minSize
      && cropWidth * cropHeight >= width * height * minAreaShare
    );
  }

  function restoreInterruptedCrop(previousBBox) {
    return Array.isArray(previousBBox) ? previousBBox.slice() : null;
  }

  return {
    canvasPointToImage,
    normalizeCrop,
    restoreInterruptedCrop,
    validateCrop,
  };
});
