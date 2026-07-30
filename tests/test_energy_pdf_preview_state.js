"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");
const recognitionState = require(
  "../static/energy/pdf_recognition_state.js"
);

const html = fs.readFileSync("templates/energy.html", "utf8");
const prepareFunctionMatch = html.match(
  /function pdfPrepareSessionMatchesCurrent\([\s\S]*?\n        \}/
);
const functionMatch = html.match(
  /function pdfPreviewRequestMatchesCurrent\([\s\S]*?\n        \}/
);

assert.ok(
  prepareFunctionMatch,
  "energy template must define pdfPrepareSessionMatchesCurrent"
);
assert.ok(
  functionMatch,
  "energy template must define pdfPreviewRequestMatchesCurrent"
);

const sandbox = {};
vm.runInNewContext(
  `${prepareFunctionMatch[0]};
  ${functionMatch[0]};
  prepareMatcher = pdfPrepareSessionMatchesCurrent;
  previewMatcher = pdfPreviewRequestMatchesCurrent;`,
  sandbox
);
const prepareMatchesCurrent = sandbox.prepareMatcher;
const previewMatchesCurrent = sandbox.previewMatcher;

assert.strictEqual(
  prepareMatchesCurrent(21, 22),
  false,
  "a prepare response from the previous upload session must not update state"
);
assert.strictEqual(
  prepareMatchesCurrent(22, 22),
  true,
  "the latest upload prepare response may update state"
);

const pageTwoRequest = {
  sessionId: 8,
  requestId: 14,
  uploadToken: "upload-a",
  pageNumber: "2",
};
const pageThreeRequest = {
  sessionId: 8,
  requestId: 15,
  uploadToken: "upload-a",
  pageNumber: "3",
};

assert.strictEqual(
  previewMatchesCurrent(pageTwoRequest, 8, 15, "upload-a", "3"),
  false,
  "an older page response must not replace the latest page"
);
assert.strictEqual(
  previewMatchesCurrent(pageThreeRequest, 8, 15, "upload-a", "3"),
  true,
  "the latest page response may update preview state"
);
assert.strictEqual(
  previewMatchesCurrent(pageThreeRequest, 9, 16, "upload-b", "1"),
  false,
  "a response from the previous upload session must not update state"
);
assert.strictEqual(
  previewMatchesCurrent(pageThreeRequest, 8, 16, "upload-a", "3"),
  false,
  "a stale image load callback must fail after a newer preview starts"
);

const behaviorFailures = [];

function behavior(name, run) {
  try {
    run();
  } catch (error) {
    behaviorFailures.push(`${name}: ${error.message}`);
  }
}

function selection(overrides = {}) {
  return {
    uploadSessionId: 8,
    uploadToken: "upload-a",
    pageNumber: "2",
    mode: "crop_region",
    cropBBox: [100, 50, 500, 350],
    previewSize: [800, 600],
    ...overrides,
  };
}

function fakeClassList(initial = []) {
  const names = new Set(initial);
  return {
    add(name) {
      names.add(name);
    },
    contains(name) {
      return names.has(name);
    },
  };
}

behavior("selection fingerprints are immutable snapshots", () => {
  assert.strictEqual(typeof recognitionState.createSelectionFingerprint, "function");
  const input = selection();
  const fingerprint = recognitionState.createSelectionFingerprint(input);

  input.cropBBox[0] = 999;
  input.previewSize[0] = 999;

  assert.deepStrictEqual(fingerprint.cropBBox, [100, 50, 500, 350]);
  assert.deepStrictEqual(fingerprint.previewSize, [800, 600]);
  assert.strictEqual(Object.isFrozen(fingerprint), true);
  assert.strictEqual(Object.isFrozen(fingerprint.cropBBox), true);
  assert.strictEqual(Object.isFrozen(fingerprint.previewSize), true);
});

behavior("selection invalidation clears visible and derived results", () => {
  assert.strictEqual(
    typeof recognitionState.invalidateDerivedRecognitionState,
    "function"
  );
  const derived = {
    aiResultData: { id: "old-ai" },
    energyResultData: { id: "old-energy" },
    calibrationPoints: [[1, 2]],
  };
  const results = { classList: fakeClassList() };
  const originalImage = {
    src: "old-original",
    removeAttribute(name) {
      if (name === "src") this.src = "";
    },
  };
  const overlayImage = {
    src: "old-overlay",
    removeAttribute(name) {
      if (name === "src") this.src = "";
    },
  };
  const scaleInput = { value: "0.0123" };
  const areaInput = { value: "456.7" };
  const wallLengthInput = { value: "89.0" };
  const windowLengthInput = { value: "12.0" };
  const calculateButton = { disabled: false };
  const calibrationPanel = { classList: fakeClassList() };

  recognitionState.invalidateDerivedRecognitionState(derived, {
    results,
    originalImage,
    overlayImage,
    scaleInput,
    areaInput,
    wallLengthInput,
    windowLengthInput,
    calculateButton,
    calibrationPanel,
  });

  assert.strictEqual(derived.aiResultData, null);
  assert.strictEqual(derived.energyResultData, null);
  assert.deepStrictEqual(derived.calibrationPoints, []);
  assert.strictEqual(results.classList.contains("hidden"), true);
  assert.strictEqual(originalImage.src, "");
  assert.strictEqual(overlayImage.src, "");
  assert.strictEqual(scaleInput.value, "");
  assert.strictEqual(areaInput.value, "");
  assert.strictEqual(wallLengthInput.value, "");
  assert.strictEqual(windowLengthInput.value, "");
  assert.strictEqual(calculateButton.disabled, true);
  assert.strictEqual(calibrationPanel.classList.contains("hidden"), true);
});

behavior("page, mode, and crop mutations reject an older response", () => {
  assert.strictEqual(
    typeof recognitionState.createRecognitionRequestState,
    "function"
  );
  for (const changedSelection of [
    selection({
      uploadSessionId: 9,
      uploadToken: "upload-b",
      pageNumber: "1",
    }),
    selection({ pageNumber: "3" }),
    selection({ mode: "full_page", cropBBox: null }),
    selection({ cropBBox: [120, 70, 520, 370] }),
  ]) {
    const requests = recognitionState.createRecognitionRequestState();
    assert.strictEqual(requests.isInFlight(), false);
    const controller = {
      aborted: false,
      abort() {
        this.aborted = true;
      },
    };
    const oldRequest = requests.begin(selection(), controller).request;
    assert.strictEqual(requests.isInFlight(), true);
    requests.invalidate();

    assert.strictEqual(controller.aborted, true);
    assert.strictEqual(requests.isInFlight(), false);
    assert.strictEqual(requests.canCommit(oldRequest, changedSelection), false);
  }
});

behavior("only the current request owns loader completion", () => {
  const requests = recognitionState.createRecognitionRequestState();
  const oldRequest = requests.begin(selection(), { abort() {} }).request;
  const invalidation = requests.invalidate();
  const currentSelection = selection({ pageNumber: "3" });
  const currentRequest = requests.begin(
    currentSelection,
    { abort() {} }
  ).request;

  assert.strictEqual(invalidation.hideLoader, true);
  assert.strictEqual(
    requests.finish(oldRequest, currentSelection).hideLoader,
    false,
    "the stale request must not hide a newer request's loader"
  );
  assert.strictEqual(
    requests.finish(currentRequest, currentSelection).hideLoader,
    true,
    "the current matching request must release its own loader"
  );
});

behavior("normal recognition controls stay disabled in flight", () => {
  assert.strictEqual(
    typeof recognitionState.deriveControlDisabledState,
    "function"
  );
  const disabled = recognitionState.deriveControlDisabledState({
    inFlight: true,
    canRecognize: true,
    hasCrop: true,
  });

  for (const control of [
    "upload",
    "page",
    "reload",
    "mode",
    "crop",
    "recognize",
    "clearCrop",
  ]) {
    assert.strictEqual(
      disabled[control],
      true,
      `${control} must remain disabled during recognition`
    );
  }
});

behavior("preview loading prevents recognition from overlapping its loader", () => {
  const disabled = recognitionState.deriveControlDisabledState({
    inFlight: false,
    selectionBusy: true,
    canRecognize: true,
    hasCrop: true,
  });

  assert.strictEqual(
    disabled.recognize,
    true,
    "recognition must stay disabled until the owning preview request finishes"
  );
});

behavior("accepted recognition is bound to the matching selection", () => {
  const requests = recognitionState.createRecognitionRequestState();
  const selected = selection();
  const request = requests.begin(selected, { abort() {} }).request;

  assert.strictEqual(requests.accept(request, selected), true);
  assert.strictEqual(requests.hasAccepted(selected), true);
  assert.strictEqual(
    requests.hasAccepted(selection({ pageNumber: "3" })),
    false
  );
});

behavior("manual calibration requests are invalidated with their selection", () => {
  assert.strictEqual(
    typeof recognitionState.createSelectionBoundRequestState,
    "function"
  );
  const requests = recognitionState.createSelectionBoundRequestState();
  const controller = {
    aborted: false,
    abort() {
      this.aborted = true;
    },
  };
  const oldRequest = requests.begin(selection(), controller);
  requests.invalidate();

  assert.strictEqual(controller.aborted, true);
  assert.strictEqual(
    requests.canCommit(oldRequest, selection({ pageNumber: "3" })),
    false,
    "a calibration response from the old selection must not update state"
  );

  const currentSelection = selection({ pageNumber: "3" });
  const currentRequest = requests.begin(currentSelection, { abort() {} });
  assert.strictEqual(
    requests.canCommit(currentRequest, currentSelection),
    true
  );
  assert.strictEqual(
    requests.finish(currentRequest, currentSelection),
    true
  );
});

behavior("the energy template wires the executable state helper", () => {
  assert.ok(
    html.includes('<script src="/energy/pdf_recognition_state.js"></script>'),
    "energy template must load the recognition state helper"
  );
  assert.ok(
    html.includes("EnergyPdfRecognitionState.createRecognitionRequestState()"),
    "energy template must use the executable request-state helper"
  );
  assert.ok(
    html.includes("AnnotationCropRegion.restoreInterruptedCrop(pdfCropBeforeDrag)"),
    "pointer cancellation must restore the crop captured before dragging"
  );
  assert.ok(
    html.includes("pdfRecognitionRequests.canCommit("),
    "recognition responses must pass the executable ownership guard"
  );
  assert.ok(
    html.includes("EnergyPdfRecognitionState.invalidateDerivedRecognitionState("),
    "selection mutation must use the tested derived-state invalidator"
  );
  assert.ok(
    html.includes("EnergyPdfRecognitionState.deriveControlDisabledState("),
    "the template controls must use the tested in-flight disabled state"
  );
  assert.ok(
    html.includes("EnergyPdfRecognitionState.createSelectionBoundRequestState()"),
    "manual calibration must use the tested selection-bound request state"
  );
});

if (behaviorFailures.length) {
  throw new Error(`recognition state behavior failures:\n${behaviorFailures.join("\n")}`);
}

console.log("energy PDF preview state tests passed");
