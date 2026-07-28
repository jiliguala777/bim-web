(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const palette = {
    0: [0, 0, 0, 0],
    1: [231, 76, 60, 205],
    2: [52, 152, 219, 225],
    3: [46, 204, 113, 225],
  };
  const state = {
    projects: [],
    project: null,
    pages: [],
    page: null,
    regions: [],
    region: null,
    view: "render",
    background: null,
    mask: null,
    width: 0,
    height: 0,
    selectedClass: 1,
    brushModelPx: 6,
    drawing: false,
    panning: false,
    spaceDown: false,
    lastPoint: null,
    scale: 1,
    offsetX: 24,
    offsetY: 24,
    history: [],
    historyIndex: -1,
    autosaveTimer: null,
    saving: null,
    dirty: false,
    editRevision: 0,
    pageGeneration: 0,
    preannotating: false,
    preparing: false,
    editingReady: false,
    cropMode: false,
    cropping: false,
    cropStart: null,
    cropBox: null,
  };

  const imageCanvas = $("#image-canvas");
  const imageContext = imageCanvas.getContext("2d", { alpha: false });
  const maskCanvas = $("#mask-canvas");
  const maskContext = maskCanvas.getContext("2d");
  const viewport = $("#viewport");
  const transform = $("#canvas-transform");

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    const type = response.headers.get("content-type") || "";
    const body = type.includes("application/json") ? await response.json() : null;
    if (!response.ok) throw new Error(body?.error || `请求失败 (${response.status})`);
    return body;
  }

  function log(message, details = "") {
    const time = new Date().toLocaleTimeString();
    $("#event-log").textContent = `[${time}] ${message}${details ? `\n${details}` : ""}`;
  }

  function setSaveState(label, kind = "idle") {
    const target = $("#save-state");
    target.textContent = label;
    target.dataset.state = kind;
  }

  function setPreparing(active) {
    state.preparing = active;
    const button = $("#prepare-page");
    button.textContent = active
      ? "\u6b63\u5728\u51c6\u5907\uff0c\u8bf7\u52ff\u91cd\u590d\u70b9\u51fb\u2026"
      : "\u51c6\u5907\u6240\u9009\u9875\u9762";
    $("#project-select").disabled = active;
    $("#page-select").disabled = active || !state.project;
    button.disabled = active || !state.project || !$("#page-select").value;
  }

  function setEditingReady(ready) {
    state.editingReady = Boolean(ready);
    const enabled = state.editingReady && !state.preannotating && !state.cropMode;
    maskCanvas.style.pointerEvents = (enabled || state.cropMode) ? "" : "none";
    setControls(enabled);
    if (enabled) updateHistoryButtons();
    else {
      $("#undo").disabled = true;
      $("#redo").disabled = true;
    }
  }

  function showWarnings(warnings = []) {
    $("#warnings").innerHTML = warnings.map((item) => `<div>⚠ ${escapeHtml(item)}</div>`).join("");
  }

  function escapeHtml(value) {
    const node = document.createElement("span");
    node.textContent = value;
    return node.innerHTML;
  }

  function currentBase() {
    const pageBase = `/api/projects/${encodeURIComponent(state.project.project_id)}/pages/${state.page.page_number}`;
    return state.region
      ? `${pageBase}/regions/${encodeURIComponent(state.region.region_id)}`
      : pageBase;
  }

  function currentTarget() {
    return state.region || state.page;
  }

  function currentPreparation() {
    return currentTarget()?.preparation || null;
  }

  function hasConfirmedTarget() {
    return state.pages.some((page) => (
      page.status === "confirmed"
      || Object.values(page.regions || {}).some((region) => region.status === "confirmed")
    ));
  }

  async function refreshProjects(selectId = null) {
    const body = await api("/api/projects");
    state.projects = body.projects;
    $("#project-select").innerHTML = '<option value="">请选择</option>' + state.projects
      .map((project) => `<option value="${escapeHtml(project.project_id)}">${escapeHtml(project.name)} · ${project.page_count} 页</option>`)
      .join("");
    if (selectId) {
      $("#project-select").value = selectId;
      await selectProject(selectId);
    }
  }

  async function selectProject(projectId) {
    if (state.dirty) await saveMask(true);
    else if (state.saving) await state.saving;
    const generation = state.pageGeneration + 1;
    state.pageGeneration = generation;
    state.project = state.projects.find((project) => project.project_id === projectId) || null;
    state.page = null;
    state.region = null;
    state.regions = [];
    clearCanvas();
    if (!state.project) {
      $("#page-select").disabled = true;
      $("#prepare-page").disabled = true;
      return;
    }
    const body = await api(`/api/projects/${encodeURIComponent(projectId)}/pages`);
    if (
      generation !== state.pageGeneration
      || state.project?.project_id !== projectId
    ) return;
    state.pages = body.pages;
    $("#page-select").innerHTML = '<option value="">请选择</option>' + state.pages
      .map((page) => `<option value="${page.page_number}">第 ${page.page_number} 页 · ${statusText(page.status)}</option>`)
      .join("");
    $("#page-select").disabled = state.preparing;
    $("#prepare-page").disabled = (
      state.preparing || !$("#page-select").value
    );
    $("#export").disabled = !hasConfirmedTarget();
    renderPageList();
    renderRegionList();
  }

  function statusText(status) {
    return {
      unprepared: "未准备",
      prepared: "已准备",
      draft: "草稿",
      preannotated: "预标注",
      confirmed: "已确认",
    }[status] || status;
  }

  function renderPageList() {
    $("#page-list").innerHTML = state.pages.map((page) =>
      `<div class="page-chip" data-status="${escapeHtml(page.status)}"><span>第 ${page.page_number} 页</span><span>${statusText(page.status)}</span></div>`
    ).join("") || '<p class="empty">没有页面</p>';
  }

  function renderRegionList() {
    $("#target-full-page").disabled = !state.page?.preparation;
    $("#start-crop").disabled = !state.page?.preparation;
    $("#target-full-page").classList.toggle("active", Boolean(state.page && !state.region));
    $("#region-list").innerHTML = state.regions.length
      ? state.regions.map((region) => (
        `<button type="button" class="region-chip${state.region?.region_id === region.region_id ? " active" : ""}" `
        + `data-region-id="${escapeHtml(region.region_id)}" data-status="${escapeHtml(region.status)}">`
        + `<span>${escapeHtml(region.name)}</span><span>${statusText(region.status)}</span></button>`
      )).join("")
      : '<p class="empty">当前页暂无裁剪区域</p>';
    $$(".region-chip").forEach((button) => button.addEventListener("click", () => {
      selectRegion(button.dataset.regionId).catch(handleError);
    }));
  }

  function setCurrentPageStatus(status) {
    if (!state.page) return;
    if (state.region) {
      state.region.status = status;
      const stored = state.page.regions?.[state.region.region_id];
      if (stored) stored.status = status;
      $("#page-status").textContent = (
        `第 ${state.page.page_number} 页 / ${state.region.name} · ${statusText(status)}`
      );
      renderRegionList();
    } else {
      state.page.status = status;
      const storedPage = state.pages.find((page) => page.page_number === state.page.page_number);
      if (storedPage) storedPage.status = status;
      const option = [...$("#page-select").options]
        .find((item) => Number(item.value) === state.page.page_number);
      if (option) option.textContent = `第 ${state.page.page_number} 页 · ${statusText(status)}`;
      $("#page-status").textContent = `第 ${state.page.page_number} 页 · ${statusText(status)}`;
    }
    $("#export").disabled = !hasConfirmedTarget();
    renderPageList();
  }

  async function selectPage(pageNumber) {
    if (state.dirty) await saveMask(true);
    else if (state.saving) await state.saving;
    const generation = state.pageGeneration + 1;
    state.pageGeneration = generation;
    state.page = state.pages.find((page) => page.page_number === Number(pageNumber)) || null;
    state.region = null;
    state.regions = Object.values(state.page?.regions || {});
    cancelCropMode();
    setEditingReady(false);
    renderRegionList();
    if (!state.page) return clearCanvas();
    if (!state.page.preparation) {
      clearCanvas();
      log("该页尚未准备，请点击“准备所选页面”");
      return;
    }
    await loadCurrentTarget(generation);
  }

  async function selectRegion(regionId) {
    if (!state.page?.preparation) return;
    if (state.dirty) await saveMask(true);
    else if (state.saving) await state.saving;
    const generation = state.pageGeneration + 1;
    state.pageGeneration = generation;
    state.region = state.regions.find((region) => region.region_id === regionId) || null;
    cancelCropMode();
    renderRegionList();
    await loadCurrentTarget(generation);
  }

  async function loadCurrentTarget(generation = state.pageGeneration) {
    const target = currentTarget();
    const preparation = currentPreparation();
    if (!state.page || !target || !preparation) return clearCanvas();
    const targetKey = state.region?.region_id || "full-page";
    $("#page-status").textContent = state.region
      ? `第 ${state.page.page_number} 页 / ${state.region.name} · ${statusText(target.status)}`
      : `第 ${state.page.page_number} 页 · ${statusText(target.status)}`;
    state.width = preparation.model_input.original_size[0];
    state.height = preparation.model_input.original_size[1];
    setupCanvases();
    const requestBase = currentBase();
    const width = state.width;
    const height = state.height;
    const targetTasks = [
      loadBackground(requestBase, generation, width, height),
    ];
    if (target.current_version) {
      targetTasks.push(loadMask(requestBase, generation, width, height));
    } else {
      state.mask = new Uint8Array(width * height);
      renderMask();
      state.dirty = false;
      state.editRevision = 0;
      setSaveState("已载入", "idle");
    }
    await Promise.all(targetTasks);
    if (
      generation !== state.pageGeneration
      || (state.region?.region_id || "full-page") !== targetKey
    ) return;
    resetHistory();
    fitView();
    setEditingReady(true);
  }

  function setupCanvases() {
    for (const canvas of [imageCanvas, maskCanvas]) {
      canvas.width = state.width;
      canvas.height = state.height;
      canvas.style.width = `${state.width}px`;
      canvas.style.height = `${state.height}px`;
    }
    transform.style.width = `${state.width}px`;
    transform.style.height = `${state.height}px`;
    state.mask = null;
    $("#canvas-empty").hidden = true;
  }

  async function loadImage(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`图像载入失败 (${response.status})`);
    return createImageBitmap(await response.blob());
  }

  async function loadBackground(
    requestBase = currentBase(),
    generation = state.pageGeneration,
    width = state.width,
    height = state.height,
    view = state.view
  ) {
    const background = await loadImage(`${requestBase}/artifact/${view}`);
    if (generation !== state.pageGeneration || view !== state.view) {
      background.close();
      return false;
    }
    state.background?.close?.();
    state.background = background;
    imageContext.fillStyle = "#fff";
    imageContext.fillRect(0, 0, width, height);
    imageContext.drawImage(state.background, 0, 0, width, height);
    return true;
  }

  async function loadMask(
    requestBase = currentBase(),
    generation = state.pageGeneration,
    width = state.width,
    height = state.height
  ) {
    const response = await fetch(`${requestBase}/mask`, { cache: "no-store" });
    if (generation !== state.pageGeneration) return false;
    const loadedMask = await (
      window.AnnotationSavedMaskLoader.decodeSavedMaskResponse(
        response,
        width,
        height,
        { disableEditing: () => setEditingReady(false) }
      )
    );
    if (generation !== state.pageGeneration) return false;
    state.mask = loadedMask;
    renderMask();
    state.dirty = false;
    state.editRevision = 0;
    setSaveState("已载入", "idle");
    return true;
  }

  function renderMask() {
    maskContext.clearRect(0, 0, state.width, state.height);
    const image = maskContext.createImageData(state.width, state.height);
    for (let index = 0; index < state.mask.length; index += 1) {
      const colour = palette[state.mask[index]] || palette[0];
      const offset = index * 4;
      image.data[offset] = colour[0];
      image.data[offset + 1] = colour[1];
      image.data[offset + 2] = colour[2];
      image.data[offset + 3] = colour[3];
    }
    maskContext.putImageData(image, 0, 0);
  }

  function applyTransform() {
    transform.style.transform = `translate(${state.offsetX}px, ${state.offsetY}px) scale(${state.scale})`;
    $("#zoom-label").textContent = `${Math.round(state.scale * 100)}%`;
  }

  function fitView() {
    if (!state.width) return;
    const margin = 38;
    state.scale = Math.min(
      (viewport.clientWidth - margin * 2) / state.width,
      (viewport.clientHeight - margin * 2) / state.height,
      1
    );
    state.offsetX = (viewport.clientWidth - state.width * state.scale) / 2;
    state.offsetY = (viewport.clientHeight - state.height * state.scale) / 2;
    applyTransform();
  }

  function canvasPoint(event) {
    const bounds = viewport.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(state.width - 1, (event.clientX - bounds.left - state.offsetX) / state.scale)),
      y: Math.max(0, Math.min(state.height - 1, (event.clientY - bounds.top - state.offsetY) / state.scale)),
    };
  }

  function renderCropSelection() {
    const selection = $("#crop-selection");
    if (!state.cropBox) {
      selection.hidden = true;
      return;
    }
    const [x0, y0, x1, y1] = state.cropBox;
    selection.style.left = `${x0}px`;
    selection.style.top = `${y0}px`;
    selection.style.width = `${x1 - x0}px`;
    selection.style.height = `${y1 - y0}px`;
    selection.hidden = false;
  }

  function updateCropCreateButton() {
    const valid = (
      state.cropBox
      && window.AnnotationCropRegion.validateCrop(
        state.cropBox,
        state.width,
        state.height
      )
      && $("#crop-region-name").value.trim()
    );
    $("#create-crop-region").disabled = !valid;
  }

  function beginCropMode() {
    if (!state.page?.preparation || state.region) return;
    state.cropMode = true;
    state.cropping = false;
    state.cropStart = null;
    state.cropBox = null;
    $("#crop-controls").hidden = false;
    $("#crop-selection").hidden = true;
    viewport.dataset.cropMode = "true";
    setEditingReady(state.editingReady);
    updateCropCreateButton();
    log("请在整页图纸上拖动鼠标框选一个楼层区域");
  }

  function cancelCropMode() {
    state.cropMode = false;
    state.cropping = false;
    state.cropStart = null;
    state.cropBox = null;
    if ($("#crop-controls")) $("#crop-controls").hidden = true;
    if ($("#crop-selection")) $("#crop-selection").hidden = true;
    if (viewport) delete viewport.dataset.cropMode;
    if ($("#crop-region-name")) $("#crop-region-name").value = "";
    if ($("#create-crop-region")) $("#create-crop-region").disabled = true;
    if (state.page?.preparation) setEditingReady(state.editingReady);
  }

  function brushWidth() {
    const resizeScale = currentPreparation()?.model_input?.resize_scale || 1;
    return Math.max(1, state.brushModelPx / resizeScale);
  }

  function stamp(x, y, radius, classId) {
    const minX = Math.max(0, Math.floor(x - radius));
    const maxX = Math.min(state.width - 1, Math.ceil(x + radius));
    const minY = Math.max(0, Math.floor(y - radius));
    const maxY = Math.min(state.height - 1, Math.ceil(y + radius));
    const radiusSquared = radius * radius;
    for (let py = minY; py <= maxY; py += 1) {
      for (let px = minX; px <= maxX; px += 1) {
        if ((px - x) ** 2 + (py - y) ** 2 <= radiusSquared) {
          state.mask[py * state.width + px] = classId;
        }
      }
    }
  }

  function drawSegment(from, to) {
    const radius = brushWidth() / 2;
    const distance = Math.hypot(to.x - from.x, to.y - from.y);
    const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * .55)));
    for (let step = 0; step <= steps; step += 1) {
      const ratio = step / steps;
      stamp(
        from.x + (to.x - from.x) * ratio,
        from.y + (to.y - from.y) * ratio,
        radius,
        state.selectedClass
      );
    }
    const colour = palette[state.selectedClass];
    maskContext.save();
    maskContext.lineWidth = radius * 2;
    maskContext.lineCap = "round";
    maskContext.lineJoin = "round";
    if (state.selectedClass === 0) {
      maskContext.globalCompositeOperation = "destination-out";
      maskContext.strokeStyle = "#000";
    } else {
      maskContext.globalCompositeOperation = "source-over";
      maskContext.strokeStyle = `rgba(${colour[0]},${colour[1]},${colour[2]},${colour[3] / 255})`;
    }
    maskContext.beginPath();
    maskContext.moveTo(from.x, from.y);
    maskContext.lineTo(to.x, to.y);
    maskContext.stroke();
    maskContext.restore();
  }

  function encodeRle(values) {
    if (!values.length) return [];
    const encoded = [];
    let value = values[0];
    let count = 1;
    for (let index = 1; index < values.length; index += 1) {
      if (values[index] === value && count < 0xffffffff) count += 1;
      else {
        encoded.push(value, count);
        value = values[index];
        count = 1;
      }
    }
    encoded.push(value, count);
    return encoded;
  }

  function decodeRle(encoded) {
    const values = new Uint8Array(state.width * state.height);
    let cursor = 0;
    for (let index = 0; index < encoded.length; index += 2) {
      values.fill(encoded[index], cursor, cursor + encoded[index + 1]);
      cursor += encoded[index + 1];
    }
    return values;
  }

  function snapshot() {
    if (state.historyIndex < state.history.length - 1) state.history.splice(state.historyIndex + 1);
    state.history.push(encodeRle(state.mask));
    if (state.history.length > 50) state.history.shift();
    state.historyIndex = state.history.length - 1;
    updateHistoryButtons();
  }

  function resetHistory() {
    state.history = [];
    state.historyIndex = -1;
    snapshot();
  }

  function restoreHistory(index) {
    if (index < 0 || index >= state.history.length) return;
    state.historyIndex = index;
    state.mask = decodeRle(state.history[index]);
    renderMask();
    markDirty();
    updateHistoryButtons();
  }

  function updateHistoryButtons() {
    $("#undo").disabled = state.historyIndex <= 0;
    $("#redo").disabled = state.historyIndex >= state.history.length - 1;
  }

  function markDirty() {
    state.dirty = true;
    state.editRevision += 1;
    setSaveState("有未保存修改", "dirty");
    clearTimeout(state.autosaveTimer);
    state.autosaveTimer = setTimeout(() => saveMask(false).catch(handleError), 1500);
  }

  async function maskBlob(maskValues, width, height) {
    const temporary = document.createElement("canvas");
    temporary.width = width;
    temporary.height = height;
    const context = temporary.getContext("2d");
    const image = context.createImageData(width, height);
    for (let index = 0; index < maskValues.length; index += 1) {
      const value = maskValues[index];
      const offset = index * 4;
      image.data[offset] = value;
      image.data[offset + 1] = value;
      image.data[offset + 2] = value;
      image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    return new Promise((resolve, reject) =>
      temporary.toBlob((blob) => blob ? resolve(blob) : reject(new Error("无法生成蒙版 PNG")), "image/png")
    );
  }

  async function saveMask(force = false) {
    if (!state.page || (!state.dirty && !force)) return null;
    if (state.saving) await state.saving;
    if (!state.page || (!state.dirty && !force)) return null;
    clearTimeout(state.autosaveTimer);
    const requestBase = currentBase();
    const projectId = state.project.project_id;
    const pageNumber = state.page.page_number;
    const regionId = state.region?.region_id || null;
    const revision = state.editRevision;
    const maskSnapshot = state.mask.slice();
    const width = state.width;
    const height = state.height;
    state.saving = (async () => {
      setSaveState("保存中…", "saving");
      const form = new FormData();
      form.append(
        "mask",
        await maskBlob(maskSnapshot, width, height),
        `page-${pageNumber}.png`
      );
      form.append("author", "local-user");
      const result = await api(`${requestBase}/mask`, { method: "POST", body: form });
      const stillCurrent = (
        state.project?.project_id === projectId
        && state.page?.page_number === pageNumber
        && (state.region?.region_id || null) === regionId
      );
      if (stillCurrent) {
        window.AnnotationSavedMaskLoader.rememberCurrentVersion(
          currentTarget(),
          result
        );
        state.dirty = state.editRevision !== revision;
        setSaveState(state.dirty ? "有未保存修改" : "已保存", state.dirty ? "dirty" : "idle");
        showWarnings(result.warnings);
        setCurrentPageStatus(result.status);
        await inspectModelMask(requestBase, maskSnapshot);
      }
      return result;
    })();
    try {
      return await state.saving;
    } finally {
      state.saving = null;
    }
  }

  async function inspectModelMask(requestBase = currentBase(), fullMask = state.mask) {
    const response = await fetch(`${requestBase}/mask?space=model512`, { cache: "no-store" });
    if (!response.ok) return;
    const bitmap = await createImageBitmap(await response.blob());
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, 512, 512).data;
    const modelClasses = new Set();
    for (let index = 0; index < pixels.length; index += 4) modelClasses.add(pixels[index]);
    const fullClasses = new Set(fullMask);
    const names = { 1: "墙", 2: "窗", 3: "门" };
    const vanished = [1, 2, 3].filter((classId) => fullClasses.has(classId) && !modelClasses.has(classId));
    if (vanished.length) showWarnings([`${vanished.map((id) => names[id]).join("、")}在 512 模型输入中消失，请加粗标注`]);
  }

  function setControls(enabled) {
    for (const id of ["save", "confirm", "preannotate"]) $( `#${id}` ).disabled = !enabled;
    for (const button of $$(".class-tool")) button.disabled = !enabled;
    $("#brush-size").disabled = !enabled;
  }

  function clearCanvas() {
    cancelCropMode();
    state.width = 0;
    state.height = 0;
    state.mask = null;
    imageCanvas.width = maskCanvas.width = 0;
    $("#canvas-empty").hidden = false;
    setEditingReady(false);
    $("#page-status").textContent = "未选择页面";
  }

  function handleError(error) {
    console.error(error);
    setSaveState("操作失败", "error");
    log("操作失败", error.message);
  }

  $("#project-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const form = new FormData(event.currentTarget);
      const body = await api("/api/projects", { method: "POST", body: form });
      log("PDF 已导入", `${body.project.page_count} 页`);
      await refreshProjects(body.project.project_id);
    } catch (error) { handleError(error); }
  });

  $("#project-select").addEventListener("change", (event) => selectProject(event.target.value).catch(handleError));
  $("#page-select").addEventListener("change", (event) => {
    $("#prepare-page").disabled = state.preparing || !event.target.value;
    selectPage(event.target.value).catch(handleError);
  });

  $("#target-full-page").addEventListener("click", () => {
    selectRegion(null).catch(handleError);
  });

  $("#start-crop").addEventListener("click", async () => {
    try {
      if (state.region) await selectRegion(null);
      beginCropMode();
    } catch (error) {
      handleError(error);
    }
  });

  $("#cancel-crop").addEventListener("click", cancelCropMode);
  $("#crop-region-name").addEventListener("input", updateCropCreateButton);
  $("#create-crop-region").addEventListener("click", async () => {
    if (!state.cropBox || !state.page || state.region) return;
    const name = $("#crop-region-name").value.trim();
    if (!window.AnnotationCropRegion.validateCrop(
      state.cropBox,
      state.width,
      state.height
    )) return;
    const pageNumber = state.page.page_number;
    const pageBase = (
      `/api/projects/${encodeURIComponent(state.project.project_id)}`
      + `/pages/${pageNumber}`
    );
    try {
      $("#create-crop-region").disabled = true;
      const body = await api(`${pageBase}/regions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          crop_bbox_px: state.cropBox,
        }),
      });
      const region = body.region;
      state.page.regions ||= {};
      state.page.regions[region.region_id] = region;
      state.regions = Object.values(state.page.regions);
      cancelCropMode();
      renderRegionList();
      await selectRegion(region.region_id);
      log(
        `裁剪区域“${region.name}”已创建`,
        `来源：第 ${pageNumber} 页；范围：${region.crop_bbox_px.join(", ")}`
      );
    } catch (error) {
      handleError(error);
      updateCropCreateButton();
    }
  });

  $("#prepare-page").addEventListener("click", async () => {
    if (
      state.preparing
      || !state.project
      || !$("#page-select").value
    ) return;
    const projectId = state.project.project_id;
    const pageNumber = Number($("#page-select").value);
    setPreparing(true);
    try {
      setSaveState("页面准备中…", "saving");
      const body = await api(
        `/api/projects/${encodeURIComponent(projectId)}/pages/prepare`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ page_number: pageNumber }),
        }
      );
      const fallback = (
        body.page.preparation?.vector_analysis?.mode === "raster_fallback"
      );
      await selectProject(projectId);
      $("#page-select").value = String(pageNumber);
      await selectPage(pageNumber);
      if (fallback) {
        log(
          `第 ${pageNumber} 页准备完成`,
          "复杂 PDF 已使用栅格模式准备，可正常标注墙、窗、门。"
        );
      } else {
        log(`第 ${pageNumber} 页准备完成`);
      }
    } catch (error) {
      handleError(error);
    } finally {
      setPreparing(false);
    }
  });

  $$(".segmented button").forEach((button) => button.addEventListener("click", async () => {
    if (!currentPreparation()) return;
    state.view = button.dataset.view;
    $$(".segmented button").forEach((item) => item.classList.toggle("active", item === button));
    try { await loadBackground(); } catch (error) { handleError(error); }
  }));

  $$(".class-tool").forEach((button) => button.addEventListener("click", () => {
    state.selectedClass = Number(button.dataset.classId);
    $$(".class-tool").forEach((item) => item.classList.toggle("active", item === button));
  }));

  $("#brush-size").addEventListener("input", (event) => {
    state.brushModelPx = Number(event.target.value);
    $("#brush-output").textContent = `${state.brushModelPx} px`;
  });

  maskCanvas.addEventListener("pointerdown", (event) => {
    if (state.cropMode && !state.spaceDown && event.button === 0) {
      state.cropping = true;
      state.cropStart = canvasPoint(event);
      state.cropBox = null;
      maskCanvas.setPointerCapture(event.pointerId);
      renderCropSelection();
      updateCropCreateButton();
      return;
    }
    if (
      !state.editingReady
      || !state.mask
      || state.preannotating
      || state.spaceDown
      || event.button === 1
    ) return;
    state.drawing = true;
    state.lastPoint = canvasPoint(event);
    maskCanvas.setPointerCapture(event.pointerId);
    drawSegment(state.lastPoint, state.lastPoint);
  });
  maskCanvas.addEventListener("pointermove", (event) => {
    if (state.cropping) {
      state.cropBox = window.AnnotationCropRegion.normalizeCrop(
        state.cropStart,
        canvasPoint(event),
        state.width,
        state.height
      );
      renderCropSelection();
      updateCropCreateButton();
      return;
    }
    if (!state.drawing) return;
    const point = canvasPoint(event);
    drawSegment(state.lastPoint, point);
    state.lastPoint = point;
  });
  maskCanvas.addEventListener("pointerup", (event) => {
    if (state.cropping) {
      state.cropping = false;
      state.cropBox = window.AnnotationCropRegion.normalizeCrop(
        state.cropStart,
        canvasPoint(event),
        state.width,
        state.height
      );
      maskCanvas.releasePointerCapture(event.pointerId);
      renderCropSelection();
      updateCropCreateButton();
      const [x0, y0, x1, y1] = state.cropBox;
      log("已框选区域", `${x1 - x0} × ${y1 - y0} 像素`);
      return;
    }
    if (!state.drawing) return;
    state.drawing = false;
    maskCanvas.releasePointerCapture(event.pointerId);
    snapshot();
    markDirty();
  });

  viewport.addEventListener("wheel", (event) => {
    if (!state.width) return;
    event.preventDefault();
    const bounds = viewport.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const imageX = (x - state.offsetX) / state.scale;
    const imageY = (y - state.offsetY) / state.scale;
    const nextScale = Math.max(.05, Math.min(8, state.scale * (event.deltaY < 0 ? 1.12 : .89)));
    state.offsetX = x - imageX * nextScale;
    state.offsetY = y - imageY * nextScale;
    state.scale = nextScale;
    applyTransform();
  }, { passive: false });

  viewport.addEventListener("pointerdown", (event) => {
    if (!(state.spaceDown || event.button === 1)) return;
    state.panning = true;
    state.lastPoint = { x: event.clientX, y: event.clientY };
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!state.panning) return;
    state.offsetX += event.clientX - state.lastPoint.x;
    state.offsetY += event.clientY - state.lastPoint.y;
    state.lastPoint = { x: event.clientX, y: event.clientY };
    applyTransform();
  });
  viewport.addEventListener("pointerup", (event) => {
    if (!state.panning) return;
    state.panning = false;
    viewport.releasePointerCapture(event.pointerId);
  });

  window.addEventListener("keydown", (event) => {
    if (event.code === "Space") { state.spaceDown = true; event.preventDefault(); }
    if ("0123".includes(event.key) && !event.ctrlKey && !event.metaKey) {
      $(`.class-tool[data-class-id="${event.key}"]`)?.click();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      (event.shiftKey ? $("#redo") : $("#undo")).click();
    }
  });
  window.addEventListener("keyup", (event) => { if (event.code === "Space") state.spaceDown = false; });

  $("#undo").addEventListener("click", () => restoreHistory(state.historyIndex - 1));
  $("#redo").addEventListener("click", () => restoreHistory(state.historyIndex + 1));
  $("#fit-view").addEventListener("click", fitView);
  $("#save").addEventListener("click", () => saveMask(true).catch(handleError));

  $("#preannotate").addEventListener("click", async () => {
    let generation = state.pageGeneration;
    let requestBase = null;
    let replacedDirty = false;
    let appliedPreannotation = false;
    try {
      if (state.dirty && !confirm("当前未保存修改会被预标注替换，继续吗？")) return;
      clearTimeout(state.autosaveTimer);
      if (state.saving) await state.saving;
      replacedDirty = state.dirty;
      state.dirty = false;
      state.editRevision += 1;
      generation = state.pageGeneration;
      requestBase = currentBase();
      const width = state.width;
      const height = state.height;
      state.preannotating = true;
      setEditingReady(state.editingReady);
      $("#project-select").disabled = true;
      $("#page-select").disabled = true;
      $("#prepare-page").disabled = true;
      setSaveState("模型识别中…", "saving");
      const requestPreannotation = (allowBlank) => api(`${requestBase}/preannotate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            author: "onnx-preannotation",
            allow_blank: allowBlank,
          }),
        });
      let result;
      try {
        result = await requestPreannotation(false);
      } catch (error) {
        const useBlank = confirm(
          `ONNX 预标注失败：${error.message}\n\n是否明确改用空白蒙版开始人工标注？`
        );
        if (!useBlank) throw error;
        result = await requestPreannotation(true);
      }
      if (generation !== state.pageGeneration) return;
      window.AnnotationSavedMaskLoader.rememberCurrentVersion(
        currentTarget(),
        result
      );
      showWarnings(result.warnings);
      await loadMask(requestBase, generation, width, height);
      if (generation !== state.pageGeneration) return;
      appliedPreannotation = true;
      resetHistory();
      setCurrentPageStatus("preannotated");
      const blankCreated = result.warnings.some((warning) => warning.includes("blank mask"));
      log(
        blankCreated
          ? "ONNX 失败，已按你的确认创建空白蒙版"
          : "预标注完成，请人工修正后确认"
      );
    } catch (error) {
      if (generation === state.pageGeneration) handleError(error);
    } finally {
      if (
        !appliedPreannotation
        && replacedDirty
        && generation === state.pageGeneration
      ) {
        state.dirty = true;
        setSaveState("预标注失败，原修改仍未保存", "dirty");
        clearTimeout(state.autosaveTimer);
        state.autosaveTimer = setTimeout(
          () => saveMask(false).catch(handleError),
          1500
        );
      }
      state.preannotating = false;
      $("#project-select").disabled = state.preparing;
      $("#page-select").disabled = state.preparing || !state.project;
      $("#prepare-page").disabled = (
        state.preparing || !state.project || !$("#page-select").value
      );
      if (currentPreparation()) {
        setEditingReady(state.editingReady);
      } else {
        setEditingReady(false);
      }
    }
  });

  $("#confirm").addEventListener("click", async () => {
    try {
      await saveMask(true);
      const result = await api(`${currentBase()}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ author: "local-user" }),
      });
      window.AnnotationSavedMaskLoader.rememberCurrentVersion(
        currentTarget(),
        result
      );
      setCurrentPageStatus(result.status);
      $("#export").disabled = false;
      log("本页已确认，可导出训练数据");
    } catch (error) { handleError(error); }
  });

  $("#export").addEventListener("click", async () => {
    try {
      const body = await api(`/api/projects/${encodeURIComponent(state.project.project_id)}/export`, { method: "POST" });
      log(
        "训练数据已导出",
        window.AnnotationExportResult.formatExportDetail(body.export)
      );
    } catch (error) { handleError(error); }
  });

  window.addEventListener("resize", () => { if (state.width) fitView(); });
  window.addEventListener("beforeunload", (event) => {
    if (state.dirty) { event.preventDefault(); event.returnValue = ""; }
  });

  refreshProjects().catch(handleError);
})();
