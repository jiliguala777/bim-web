"use strict";

const assert = require("node:assert/strict");
const {
  decodeSavedMaskResponse,
  rememberCurrentVersion,
} = require("../annotation_tool/static/saved_mask_loader.js");

async function successfulSavedMaskIsDecoded() {
  let disabled = 0;
  let closed = 0;
  const pixels = new Uint8ClampedArray([
    2, 2, 2, 255,
    3, 3, 3, 255,
  ]);
  const decoded = await decodeSavedMaskResponse(
    {
      ok: true,
      status: 200,
      blob: async () => ({ type: "image/png" }),
    },
    2,
    1,
    {
      disableEditing: () => { disabled += 1; },
      createImageBitmap: async () => ({
        close: () => { closed += 1; },
      }),
      createCanvas: () => ({
        width: 0,
        height: 0,
        getContext: () => ({
          imageSmoothingEnabled: true,
          drawImage: () => {},
          getImageData: () => ({ data: pixels }),
        }),
      }),
    }
  );

  assert.deepEqual([...decoded], [2, 3]);
  assert.equal(disabled, 0);
  assert.equal(closed, 1);
}

async function savedMaskHttpErrorsRejectWithoutDecoding() {
  for (const status of [404, 500]) {
    let disabled = 0;
    let decoded = 0;
    await assert.rejects(
      () => decodeSavedMaskResponse(
        {
          ok: false,
          status,
          blob: async () => {
            decoded += 1;
            return {};
          },
        },
        2,
        1,
        {
          disableEditing: () => { disabled += 1; },
          createImageBitmap: async () => {
            decoded += 1;
            return {};
          },
          createCanvas: () => {
            decoded += 1;
            return {};
          },
        }
      ),
      (error) => (
        error instanceof Error
        && error.message.includes(String(status))
        && error.message.includes("编辑已禁用")
      )
    );
    assert.equal(disabled, 1);
    assert.equal(decoded, 0);
  }
}

function successfulMaskMutationsRememberTheirVersion() {
  const page = { current_version: null };

  assert.equal(
    rememberCurrentVersion(page, { version_id: "v0002" }),
    true
  );
  assert.equal(page.current_version, "v0002");
  assert.equal(rememberCurrentVersion(page, {}), false);
  assert.equal(page.current_version, "v0002");
}

Promise.all([
  successfulSavedMaskIsDecoded(),
  savedMaskHttpErrorsRejectWithoutDecoding(),
]).catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
successfulMaskMutationsRememberTheirVersion();
