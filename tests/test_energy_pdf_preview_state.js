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
const normalizedReportNumberMatch = html.match(
  /function normalizedReportNumber\([\s\S]*?\n        \}/
);
const normalizedReportOwnerMatch = html.match(
  /function normalizedReportOwnerUsername\([\s\S]*?\n        \}/
);
const selectionFunctionMatch = html.match(
  /function currentPdfRecognitionSelection\([\s\S]*?\n        \}/
);

assert.ok(
  prepareFunctionMatch,
  "energy template must define pdfPrepareSessionMatchesCurrent"
);
assert.ok(
  functionMatch,
  "energy template must define pdfPreviewRequestMatchesCurrent"
);
assert.ok(
  normalizedReportNumberMatch,
  "energy template must define normalizedReportNumber"
);
assert.ok(
  normalizedReportOwnerMatch,
  "energy template must normalize the active report owner"
);
assert.ok(
  selectionFunctionMatch,
  "energy template must bind recognition state to the active report identity"
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

const ownerSelectionSandbox = {
  EnergyPdfRecognitionState: recognitionState,
  activeReportOwnerUsername: "alice",
  preparedPdf: { uploadToken: "upload-a" },
  pdfUploadSessionId: 8,
  pdfRecognitionMode: "crop_region",
  pdfCropBBox: [100, 50, 500, 350],
  pdfPreviewSize: [800, 600],
  document: {
    getElementById(id) {
      return {
        value: id === "current-report-number" ? "SHARED-REPORT" : "2",
      };
    },
  },
};
vm.runInNewContext(
  `${normalizedReportNumberMatch[0]};
  ${normalizedReportOwnerMatch[0]};
  ${selectionFunctionMatch[0]};
  selectionForActiveOwner = currentPdfRecognitionSelection;`,
  ownerSelectionSandbox
);
const selectionForActiveOwner = ownerSelectionSandbox.selectionForActiveOwner;
const ownerBoundRequests = recognitionState.createRecognitionRequestState();
const aliceSelection = selectionForActiveOwner();
const aliceRequest = ownerBoundRequests.begin(aliceSelection, { abort() {} }).request;
ownerSelectionSandbox.activeReportOwnerUsername = "bob";
const bobSelection = selectionForActiveOwner();
assert.strictEqual(
  ownerBoundRequests.canCommit(aliceRequest, bobSelection),
  false,
  "the pending recognition fingerprint must include the active owner"
);
ownerBoundRequests.invalidate();
assert.strictEqual(
  ownerBoundRequests.canCommit(aliceRequest, bobSelection),
  false,
  "switching owner with the same report number must invalidate pending recognition"
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

function inlineEnergyScript() {
  return [...html.matchAll(/<script(?:[^>]*)>([\s\S]*?)<\/script>/g)]
    .map((match) => match[1])
    .find(Boolean);
}

class FakeElement {
  constructor(tagName, innerHtmlWrites) {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this._textContent = "";
    this._innerHTML = "";
    this.innerHtmlWrites = innerHtmlWrites;
    this.classList = { add() {}, remove() {}, toggle() {} };
  }

  set innerHTML(value) {
    this._innerHTML = String(value);
    this.innerHtmlWrites.push(this._innerHTML);
    this.children = [];
    this._textContent = "";
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set textContent(value) {
    this._textContent = String(value);
    this.children = [];
    this._innerHTML = "";
  }

  get textContent() {
    return this._textContent + this.children.map((child) => child.textContent).join("");
  }

  set innerText(value) {
    this.textContent = value;
  }

  get innerText() {
    return this.textContent;
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  addEventListener() {}
  setAttribute() {}
  removeAttribute() {}
  querySelectorAll() { return []; }
}

function createEnergyTemplateContext({ isAdministrator = true, fetch }) {
  const innerHtmlWrites = [];
  const elements = new Proxy({}, {
    get(target, key) {
      if (!target[key]) target[key] = new FakeElement("div", innerHtmlWrites);
      return target[key];
    },
  });
  const document = {
    body: { dataset: { isAdministrator: String(isAdministrator) } },
    addEventListener() {},
    getElementById(id) { return elements[id]; },
    querySelectorAll() { return []; },
    createElement(tagName) { return new FakeElement(tagName, innerHtmlWrites); },
  };
  const context = {
    EnergyPdfRecognitionState: recognitionState,
    URLSearchParams,
    AbortController,
    FormData,
    Chart: function Chart() {},
    alert() {},
    console,
    document,
    fetch,
    window: { addEventListener() {}, scrollTo() {} },
  };
  vm.createContext(context);
  vm.runInContext(inlineEnergyScript(), context);
  return { context, elements, innerHtmlWrites };
}

async function historyCardsKeepServerValuesAsText() {
  const malicious = {
    username: '<img src=x onerror="steal()">',
    report_number: '<svg onload="steal()">',
    created_at: "2026-08-31T12:00:00Z",
    floor_area: '<iframe src="javascript:steal()">',
    eui: '<script>steal()</script>',
    total_energy: 1234,
    rating: '<a onclick="steal()">A</a>',
  };
  const harness = createEnergyTemplateContext({
    fetch: async () => ({ json: async () => [malicious] }),
  });

  await vm.runInContext("openHistoryDrawer()", harness.context);

  const container = harness.elements.historyListContainer;
  assert.strictEqual(container.children.length, 1, "one report must render one card");
  const renderedText = container.children[0].textContent;
  for (const literal of [
    malicious.username,
    malicious.report_number,
    malicious.floor_area,
    malicious.eui,
    malicious.rating,
  ]) {
    assert.ok(renderedText.includes(literal), `history value must remain literal text: ${literal}`);
  }
  assert.ok(
    harness.innerHtmlWrites.every((write) => !write.includes("steal()")),
    "server-supplied history values must never reach innerHTML",
  );

  const maliciousError = '<img src=x onerror="stealError()">';
  const errorHarness = createEnergyTemplateContext({
    fetch: async () => { throw new Error(maliciousError); },
  });
  await vm.runInContext("openHistoryDrawer()", errorHarness.context);
  assert.ok(
    errorHarness.elements.historyListContainer.textContent.includes(maliciousError),
    "history errors must remain visible as literal text",
  );
  assert.ok(
    errorHarness.innerHtmlWrites.every((write) => !write.includes("stealError()")),
    "server-supplied error text must never reach innerHTML",
  );

  const ordinaryHarness = createEnergyTemplateContext({
    isAdministrator: false,
    fetch: async () => ({
      json: async () => [{
        ...malicious,
        username: "hidden-owner",
        report_number: "USER-REPORT",
        floor_area: 120,
        eui: 45,
        rating: "A",
      }],
    }),
  });
  ordinaryHarness.context.selected = [];
  vm.runInContext(
    "loadReportDetails = (reportNumber, ownerUsername) => selected.push([reportNumber, ownerUsername]);",
    ordinaryHarness.context,
  );
  await vm.runInContext("openHistoryDrawer()", ordinaryHarness.context);
  const ordinaryCard = ordinaryHarness.elements.historyListContainer.children[0];
  assert.ok(ordinaryCard.textContent.includes("USER-REPORT"));
  assert.ok(!ordinaryCard.textContent.includes("hidden-owner"));
  ordinaryCard.onclick();
  assert.strictEqual(ordinaryHarness.context.selected[0].join("\0"), "USER-REPORT\0");
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

async function staleReportDetailsCannotReplaceNewerSelection() {
  const requests = [];
  const harness = createEnergyTemplateContext({
    fetch(url) {
      const response = deferred();
      requests.push({ url, response });
      return response.promise;
    },
  });
  const tracking = { displays: [], identities: [], steps: [], alerts: [] };
  harness.context.tracking = tracking;
  vm.runInContext(`
    closeHistoryDrawer = () => {};
    showLoader = () => {};
    hideLoader = () => {};
    alert = message => tracking.alerts.push(message);
    setActiveReportIdentity = (nextReportNumber, nextOwnerUsername) => {
      reportNumber = String(nextReportNumber || '').trim();
      activeReportOwnerUsername = nextOwnerUsername || null;
      tracking.identities.push([activeReportOwnerUsername, reportNumber]);
    };
    normalizeDetailedEnvelope = () => ({});
    initializeDetailedEnvelope = () => {};
    renderDetailedOrientationCards = () => {};
    setCalculationMode = () => {};
    updateBoundaryControlState = () => {};
    inferCalculationEnabled = () => true;
    heatingSystemEfficiencyDefault = () => 1;
    updateCalculationScopeControls = () => {};
    updatePhysicalLengths = () => {};
    updateAnnualUsePreview = () => {};
    updateDesignTemperatureNotice = () => {};
    displayResults = data => tracking.displays.push(data.marker);
    gotoStep = step => tracking.steps.push(step);
  `, harness.context);

  const loadA = vm.runInContext("loadReportDetails('SHARED-REPORT', 'alice')", harness.context);
  const loadB = vm.runInContext("loadReportDetails('SHARED-REPORT', 'bob')", harness.context);
  assert.deepStrictEqual(
    requests.map((request) => request.url),
    [
      "/energy/report/SHARED-REPORT?owner_username=alice",
      "/energy/report/SHARED-REPORT?owner_username=bob",
    ],
  );

  requests[1].response.resolve({
    json: async () => ({
      report_number: "SHARED-REPORT",
      username: "bob",
      params: {},
      results: { marker: "bob result" },
    }),
  });
  await loadB;
  requests[0].response.resolve({
    json: async () => ({
      report_number: "SHARED-REPORT",
      username: "alice",
      params: {},
      results: { marker: "alice result" },
    }),
  });
  await loadA;

  assert.strictEqual(harness.elements["current-report-number"].value, "SHARED-REPORT");
  assert.deepStrictEqual(tracking.displays, ["bob result"]);
  assert.deepStrictEqual(tracking.steps, [4]);
  assert.strictEqual(tracking.identities.at(-1).join("\0"), "bob\0SHARED-REPORT");
  assert.deepStrictEqual(tracking.alerts, []);

  const loadC = vm.runInContext("loadReportDetails('REPORT-C', 'carol')", harness.context);
  requests[2].response.resolve({
    json: async () => ({ error: "current detail failure" }),
  });
  await loadC;
  assert.strictEqual(tracking.alerts.length, 1);
  assert.ok(tracking.alerts[0].includes("current detail failure"));
}

async function main() {
  const failures = [...behaviorFailures];
  for (const [name, run] of [
    ["history cards keep server values as text", historyCardsKeepServerValuesAsText],
    ["stale report details cannot replace newer selection", staleReportDetailsCannotReplaceNewerSelection],
  ]) {
    try {
      await run();
    } catch (error) {
      failures.push(`${name}: ${error.message}`);
    }
  }
  if (failures.length) {
    throw new Error(`energy template behavior failures:\n${failures.join("\n")}`);
  }
  console.log("energy PDF preview state tests passed");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
