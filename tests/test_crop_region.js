"use strict";

const assert = require("assert");
const crop = require("../static/energy/crop_region.js");

assert.deepStrictEqual(
  crop.normalizeCrop(
    { x: 300.4, y: 250.8 },
    { x: 100.2, y: 50.1 },
    400,
    300
  ),
  [100, 50, 301, 251]
);

assert.deepStrictEqual(
  crop.canvasPointToImage(
    { x: 250, y: 180 },
    { offsetX: 50, offsetY: 30, scale: 2, width: 400, height: 300 }
  ),
  { x: 100, y: 75 }
);

assert.deepStrictEqual(
  crop.normalizeCrop(
    { x: -20, y: -10 },
    { x: 450, y: 350 },
    400,
    300
  ),
  [0, 0, 400, 300]
);

assert.deepStrictEqual(
  crop.normalizeCrop(
    { x: 450, y: 350 },
    { x: -20, y: -10 },
    400,
    300
  ),
  [0, 0, 400, 300]
);

assert.deepStrictEqual(
  crop.canvasPointToImage(
    { x: -50, y: 900 },
    { offsetX: 50, offsetY: 30, scale: 2, width: 400, height: 300 }
  ),
  { x: 0, y: 300 }
);

assert.strictEqual(crop.validateCrop([0, 0, 128, 128], 400, 300), true);
assert.strictEqual(crop.validateCrop([0, 0, 127, 200], 400, 300), false);
assert.strictEqual(crop.validateCrop([0, 0, 400, 2], 400, 300), false);
assert.strictEqual(crop.validateCrop([0, 0, 401, 200], 400, 300), false);
assert.strictEqual(
  crop.validateCrop([0, 0, 199, 100], 1000, 2000, 1, 0.01),
  false,
  "a crop just below 1% of the preview area must be rejected"
);
assert.strictEqual(
  crop.validateCrop([0, 0, 200, 100], 1000, 2000, 1, 0.01),
  true,
  "a crop exactly 1% of the preview area must be accepted"
);

assert.strictEqual(
  typeof crop.restoreInterruptedCrop,
  "function",
  "crop helper must expose interrupted-drag restoration behavior"
);
const cropBeforeInterruptedDrag = [10, 20, 210, 220];
const restoredInterruptedCrop = crop.restoreInterruptedCrop(cropBeforeInterruptedDrag);
assert.deepStrictEqual(
  restoredInterruptedCrop,
  [10, 20, 210, 220],
  "pointer cancellation must restore the crop that existed before dragging"
);
assert.notStrictEqual(
  restoredInterruptedCrop,
  cropBeforeInterruptedDrag,
  "the restored crop must not alias mutable drag state"
);
assert.strictEqual(
  crop.restoreInterruptedCrop(null),
  null,
  "pointer cancellation with no prior crop must clear the partial selection"
);

console.log("crop_region tests passed");
