"use strict";

const assert = require("assert");
const crop = require("../annotation_tool/static/crop_region.js");

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

assert.strictEqual(crop.validateCrop([0, 0, 128, 128], 400, 300), true);
assert.strictEqual(crop.validateCrop([0, 0, 127, 200], 400, 300), false);
assert.strictEqual(crop.validateCrop([0, 0, 400, 2], 400, 300), false);
assert.strictEqual(crop.validateCrop([0, 0, 401, 200], 400, 300), false);

console.log("crop_region tests passed");
