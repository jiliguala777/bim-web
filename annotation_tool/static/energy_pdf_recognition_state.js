(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.EnergyPdfRecognitionState = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function immutableArray(value) {
    return Array.isArray(value) ? Object.freeze(value.slice()) : null;
  }

  function createSelectionFingerprint(selection) {
    const source = selection || {};
    return Object.freeze({
      uploadSessionId:
        source.uploadSessionId == null ? null : Number(source.uploadSessionId),
      uploadToken:
        source.uploadToken == null ? null : String(source.uploadToken),
      pageNumber:
        source.pageNumber == null ? null : String(source.pageNumber),
      mode: source.mode === "crop_region" ? "crop_region" : "full_page",
      cropBBox: immutableArray(source.cropBBox),
      previewSize: immutableArray(source.previewSize),
    });
  }

  function arraysEqual(left, right) {
    if (left === right) return true;
    if (!left || !right || left.length !== right.length) return false;
    return left.every((value, index) => value === right[index]);
  }

  function fingerprintsEqual(left, right) {
    if (!left || !right) return false;
    return (
      left.uploadSessionId === right.uploadSessionId
      && left.uploadToken === right.uploadToken
      && left.pageNumber === right.pageNumber
      && left.mode === right.mode
      && arraysEqual(left.cropBBox, right.cropBBox)
      && arraysEqual(left.previewSize, right.previewSize)
    );
  }

  function invalidateDerivedRecognitionState(derivedState, elements) {
    derivedState.aiResultData = null;
    derivedState.energyResultData = null;
    derivedState.calibrationPoints = [];

    const view = elements || {};
    view.results?.classList.add("hidden");
    view.calibrationPanel?.classList.add("hidden");
    for (const image of [view.originalImage, view.overlayImage]) {
      image?.removeAttribute("src");
    }
    for (const input of [
      view.scaleInput,
      view.areaInput,
      view.wallLengthInput,
      view.windowLengthInput,
    ]) {
      if (input) input.value = "";
    }
    if (view.calculateButton) view.calculateButton.disabled = true;
    if (view.closureSummary) view.closureSummary.innerText = "";
    if (view.calibrationStatus) view.calibrationStatus.innerText = "";
    if (view.calibrationCanvas) {
      view.calibrationCanvas.style.pointerEvents = "none";
    }
    return derivedState;
  }

  function deriveControlDisabledState(options) {
    const state = options || {};
    const inFlight = Boolean(state.inFlight);
    const selectionBusy = Boolean(state.selectionBusy);
    return Object.freeze({
      upload: inFlight,
      page: inFlight,
      reload: inFlight,
      mode: inFlight,
      crop: inFlight,
      recognize: inFlight || selectionBusy || !state.canRecognize,
      clearCrop: inFlight || !state.hasCrop,
    });
  }

  function createSelectionBoundRequestState() {
    let generation = 0;
    let active = null;

    function cancelActive() {
      if (active?.controller?.abort) {
        active.controller.abort();
      }
      active = null;
    }

    function invalidate() {
      generation += 1;
      cancelActive();
    }

    function begin(selection, controller) {
      cancelActive();
      generation += 1;
      const request = Object.freeze({
        requestId: generation,
        fingerprint: createSelectionFingerprint(selection),
      });
      active = { request, controller };
      return request;
    }

    function canCommit(request, currentSelection) {
      if (!request || !active) return false;
      return (
        request.requestId === generation
        && active.request.requestId === request.requestId
        && fingerprintsEqual(
          request.fingerprint,
          createSelectionFingerprint(currentSelection)
        )
      );
    }

    function finish(request, currentSelection) {
      if (!canCommit(request, currentSelection)) return false;
      active = null;
      return true;
    }

    return Object.freeze({
      begin,
      canCommit,
      finish,
      invalidate,
    });
  }

  function createRecognitionRequestState() {
    let generation = 0;
    let active = null;
    let acceptedFingerprint = null;
    let loaderOwnerRequestId = null;

    function cancelActive() {
      const cancelled = active;
      const hideLoader = Boolean(
        cancelled
        && loaderOwnerRequestId === cancelled.request.requestId
      );
      if (cancelled?.controller?.abort) {
        cancelled.controller.abort();
      }
      active = null;
      loaderOwnerRequestId = null;
      return Object.freeze({
        cancelledRequest: cancelled?.request || null,
        hideLoader,
      });
    }

    function invalidate() {
      generation += 1;
      acceptedFingerprint = null;
      return cancelActive();
    }

    function begin(selection, controller) {
      const invalidation = cancelActive();
      generation += 1;
      acceptedFingerprint = null;
      const request = Object.freeze({
        requestId: generation,
        fingerprint: createSelectionFingerprint(selection),
      });
      active = { request, controller };
      loaderOwnerRequestId = request.requestId;
      return Object.freeze({
        request,
        cancelledRequest: invalidation.cancelledRequest,
        hideLoader: invalidation.hideLoader,
      });
    }

    function canCommit(request, currentSelection) {
      if (!request || !active) return false;
      return (
        request.requestId === generation
        && active.request.requestId === request.requestId
        && fingerprintsEqual(
          request.fingerprint,
          createSelectionFingerprint(currentSelection)
        )
      );
    }

    function accept(request, currentSelection) {
      if (!canCommit(request, currentSelection)) return false;
      acceptedFingerprint = request.fingerprint;
      return true;
    }

    function finish(request, currentSelection) {
      if (!canCommit(request, currentSelection)) {
        return Object.freeze({ hideLoader: false });
      }
      const hideLoader = loaderOwnerRequestId === request.requestId;
      active = null;
      loaderOwnerRequestId = null;
      return Object.freeze({ hideLoader });
    }

    function hasAccepted(currentSelection) {
      return fingerprintsEqual(
        acceptedFingerprint,
        createSelectionFingerprint(currentSelection)
      );
    }

    function isInFlight() {
      return Boolean(active);
    }

    return Object.freeze({
      accept,
      begin,
      canCommit,
      finish,
      hasAccepted,
      invalidate,
      isInFlight,
    });
  }

  return {
    createRecognitionRequestState,
    createSelectionBoundRequestState,
    createSelectionFingerprint,
    deriveControlDisabledState,
    fingerprintsEqual,
    invalidateDerivedRecognitionState,
  };
});
