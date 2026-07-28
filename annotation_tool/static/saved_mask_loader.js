(function registerSavedMaskLoader(root, factory) {
  "use strict";

  const api = factory(root);
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.AnnotationSavedMaskLoader = api;
  }
})(
  typeof globalThis === "object" ? globalThis : this,
  (root) => {
    "use strict";

    async function decodeSavedMaskResponse(
      response,
      width,
      height,
      dependencies = {}
    ) {
      const disableEditing = dependencies.disableEditing || (() => {});
      try {
        if (!response.ok) {
          const error = new Error(
            `已保存蒙版载入失败 (${response.status})，编辑已禁用，请重试或修复该版本`
          );
          error.savedMaskLoadError = true;
          throw error;
        }
        const createBitmap = (
          dependencies.createImageBitmap || root.createImageBitmap
        );
        const createCanvas = (
          dependencies.createCanvas
          || (() => root.document.createElement("canvas"))
        );
        const bitmap = await createBitmap(await response.blob());
        try {
          const temporary = createCanvas();
          temporary.width = width;
          temporary.height = height;
          const context = temporary.getContext(
            "2d",
            { willReadFrequently: true }
          );
          context.imageSmoothingEnabled = false;
          context.drawImage(bitmap, 0, 0, width, height);
          const pixels = context.getImageData(0, 0, width, height).data;
          const loadedMask = new Uint8Array(width * height);
          for (let index = 0; index < loadedMask.length; index += 1) {
            loadedMask[index] = pixels[index * 4];
          }
          return loadedMask;
        } finally {
          bitmap.close();
        }
      } catch (error) {
        disableEditing();
        if (error.savedMaskLoadError) throw error;
        const wrapped = new Error(
          "已保存蒙版无法解码，编辑已禁用，请重试或修复该版本"
        );
        wrapped.cause = error;
        throw wrapped;
      }
    }

    function rememberCurrentVersion(page, result) {
      if (!page || !result || !result.version_id) return false;
      page.current_version = result.version_id;
      return true;
    }

    return Object.freeze({
      decodeSavedMaskResponse,
      rememberCurrentVersion,
    });
  }
);
