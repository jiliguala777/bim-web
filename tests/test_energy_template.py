import unittest
from pathlib import Path
import base64
import re
import io
import json
import os
import subprocess
import tempfile
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


class EnergyTemplateTests(unittest.TestCase):
    def test_recognition_ui_uses_closed_room_area_instead_of_image_rectangle_guess(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn('id="room-closure-summary"', html)
        self.assertIn("totalAreaPx2 * scale * scale", html)
        self.assertNotIn("total_pixels * (scale ** 2) * 0.3", html)
        self.assertIn("manual_exterior_wall_required", html)
        self.assertIn("closure_status", html)
        self.assertIn("accepted_lines_px", html)
        self.assertIn("自动连接了", html)
        self.assertIn("已闭合房间面积", html)
        self.assertNotIn("外轮廓无法由唯一一条直线自动闭合", html)

    def test_scale_ui_uses_auto_result_and_supports_manual_two_point_fallback(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn('id="scale-calibration-canvas"', html)
        self.assertIn('id="scale-calibration-panel"', html)
        self.assertIn("function startManualScaleCalibration()", html)
        self.assertIn("function saveManualScaleCalibration()", html)
        self.assertIn("fetch('/energy/scale_calibration'", html)
        self.assertIn("!hasCurrentConfirmedGeometry()", html)
        self.assertIn("!hasCurrentAiRecognitionResult()", html)
        self.assertIn('id="param-scale" class="form-control" value=""', html)
        self.assertNotIn("const scale = 0.05;", html)

    def test_manual_scale_canvas_resizes_after_becoming_visible(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        start_body = re.search(
            r"function startManualScaleCalibration\(\) \{(?P<body>.*?)\n        \}",
            html,
            re.S,
        ).group("body")

        self.assertIn("resizeScaleCalibrationCanvas()", start_body)
        self.assertLess(
            start_body.index("resizeScaleCalibrationCanvas()"),
            start_body.index("canvas.style.pointerEvents = 'auto';"),
        )
        self.assertRegex(
            html,
            r"gotoStep\(2\);\s*requestAnimationFrame\(\(\) => resizeScaleCalibrationCanvas\(\)\);",
        )

    def test_approved_onnx_model_is_the_default(self):
        server = Path("web_server_server.py").read_text(encoding="utf-8")
        segmenter = Path("floorplan_onnx.py").read_text(encoding="utf-8")

        self.assertIn("os.environ.get('ONNX_MODEL_PATH'", server)
        self.assertIn("os.path.join(MODELS_DIR, 'M2_pub_plus_user.onnx')", server)
        self.assertIn(
            'Path(__file__).parent / "models" / "M2_pub_plus_user.onnx"',
            segmenter,
        )
        self.assertNotIn("M2_DA_best.onnx", server)
        self.assertNotIn("M2_DA_best.onnx", segmenter)

    def test_detailed_wall_modal_combines_base_and_insulation_u_values(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn("function updateModalCombinedWallU", html)
        self.assertIn("applyModalWallBase", html)
        self.assertIn("applyModalWallInsulation", html)
        self.assertIn("1 / baseU", html)
        self.assertIn("+ insulationR", html)
        self.assertIn("if (uValue) return 1 / uValue", html)
        self.assertIn("modal-wall-insulation-thickness", html)
        self.assertIn("(thicknessMm / 1000) / lambdaValue", html)

    def test_detailed_boundary_controls_default_off_and_multipage_pdf_notice_exists(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn('type="checkbox" id="include-roof"', html)
        self.assertIn('type="checkbox" id="include-floor"', html)
        self.assertNotIn('id="include-roof" checked', html)
        self.assertNotIn('id="include-floor" checked', html)
        self.assertIn('id="param-annual-solar-irradiation"', html)
        self.assertIn('value="150" min="0"', html)
        self.assertIn("kWh/(m²·a)", html)
        self.assertIn("多页 PDF 可在上传后选择需要识别的楼层页", html)

    def test_multipage_pdf_ui_prepares_once_then_recognizes_selected_page(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn('id="pdf-page-selection"', html)
        self.assertIn('id="pdf-page-select"', html)
        self.assertIn('id="recognize-selected-page"', html)
        self.assertIn("async function preparePdf(file)", html)
        self.assertIn("fetch('/energy/pdf_prepare'", html)
        self.assertIn("async function recognizePreparedPdf()", html)
        self.assertIn("fd.append('pdf_upload_token', preparedPdf.uploadToken)", html)
        self.assertIn("fd.append('pdf_page_number', document.getElementById('pdf-page-select').value)", html)

    def test_pdf_region_recognition_ui_exposes_preview_modes_and_crop_controls(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        for element_id in [
            "pdf-recognition-mode-full",
            "pdf-recognition-mode-crop",
            "pdf-page-preview-image",
            "pdf-crop-overlay",
            "clear-pdf-crop",
            "pdf-crop-summary",
        ]:
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', html)
        self.assertIn('<script src="/energy/crop_region.js">', html)

    def test_vector_raster_ui_exposes_an_independent_crop_preview(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        for element_id in [
            "vector-raster-selection",
            "vector-raster-preview-image",
            "vector-raster-crop-overlay",
            "vector-raster-mode-full",
            "vector-raster-mode-crop",
            "clear-vector-raster-crop",
            "recognize-vector-raster",
        ]:
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', html)
        self.assertIn("async function prepareVectorRasterCrop(file)", html)
        self.assertIn("vectorRasterRecognitionMode", html)

    def test_vector_pdf_backend_uses_isolated_debug_route_and_blocks_energy_geometry(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn("新矢量模型（矢量 PDF/图片试验版）", html)
        self.assertIn("const vectorPdfFusion = Boolean(!file && preparedPdf && vectorBackend);", html)
        self.assertIn("'/energy/vector_pdf_fusion'", html)
        self.assertIn("if (data.fusion_debug)", html)
        self.assertIn("aiResultData = null;", html)
        self.assertIn("调试结果，尚不可用于能耗计算", html)

    def test_exterior_debug_requires_confirmation_before_energy_state(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn("pendingExteriorFusion", html)
        self.assertIn("/energy/vector_pdf_exterior_confirm", html)
        self.assertIn("确认外轮廓并用于能耗计算", html)
        self.assertIn('id="exterior-review-panel"', html)
        self.assertIn('id="component-review-overlay"', html)
        self.assertIn("images.component_overlay", html)
        self.assertIn("area_px2", html)
        self.assertIn("perimeter_px", html)
        self.assertIn("door_count", html)
        self.assertIn("window_count", html)
        self.assertIn("pending_opening_count", html)
        self.assertIn("待确认开口", html)
        self.assertIn("recovered_wall_count", html)
        self.assertIn("confirmed_door_arc_count", html)
        self.assertIn("pending_door_arc_count", html)
        self.assertIn("门弧恢复墙段", html)
        self.assertIn("small_repair_count", html)
        self.assertIn("unresolved_gap_count", html)
        self.assertIn("topology_sha256: pending.topology_sha256", html)
        self.assertIn("scale_m_per_px: scale", html)
        self.assertIn("confirmed: true", html)
        self.assertIn("page_number: pending.pdf_page_number", html)
        self.assertIn("crop_bbox_page_px: pending.crop_bbox_page_px", html)
        self.assertIn("aiResultData = data.recognition;", html)

    def test_exterior_debug_keeps_component_overlay_and_supports_image_enlargement(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn('id="component-review-overlay"', html)
        self.assertIn("images.component_overlay", html)
        self.assertIn('id="recognition-image-modal"', html)
        self.assertIn("openRecognitionImageModal", html)

    def test_exterior_review_explains_footprint_guided_inference(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        for element_id in (
            "exterior-closure-method-summary",
            "exterior-inferred-edge-summary",
            "exterior-inferred-length-summary",
            "exterior-inferred-ratio-summary",
        ):
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', html)
        self.assertIn("footprint_guided_inference", html)
        self.assertIn("橙色虚线", html)
        self.assertIn("inferred_edge_count", html)
        self.assertIn("inferred_length_px", html)
        self.assertIn("inferred_perimeter_ratio", html)

    def test_exterior_height_and_repeat_controls_are_sent_to_energy_route(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        expected_inputs = {
            "param-door-height": ('min="0.1"', 'step="0.1"', 'value="2.1"'),
            "param-window-height": ('min="0.1"', 'step="0.1"', 'value="1.5"'),
            "param-door-repeat-count": ('min="1"', 'step="1"', 'value="1"'),
            "param-window-repeat-count": ('min="1"', 'step="1"'),
        }
        for element_id, fragments in expected_inputs.items():
            with self.subTest(element_id=element_id):
                input_tag = re.search(
                    rf'<input[^>]*id="{element_id}"[^>]*>', html,
                ).group(0)
                for fragment in fragments:
                    self.assertIn(fragment, input_tag)

        for field in (
            "door_height_m",
            "window_height_m",
            "door_repeat_count",
            "window_repeat_count",
        ):
            with self.subTest(field=field):
                self.assertIn(f"{field}:", html)

    def test_exterior_pending_and_confirmed_state_is_cleared_with_recognition_selection(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        clear_segment = html[
            html.index("function clearPdfRecognitionDerivedState()")
            :html.index("function invalidatePdfRecognitionSelection")
        ]

        self.assertIn("pendingExteriorFusion = null;", clear_segment)
        self.assertIn("aiResultData = derivedState.aiResultData;", clear_segment)
        self.assertIn("function hasCurrentConfirmedGeometry()", html)
        self.assertIn("exterior_topology?.confirmed === true", html)
        self.assertIn("room_topology?.load_geometry_ready === true", html)

    def test_report_identity_change_invalidates_executable_client_state(self):
        node_script = r'''
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('templates/energy.html', 'utf8');
const inlineScript = [...html.matchAll(/<script(?:[^>]*)>([\s\S]*?)<\/script>/g)]
  .map(match => match[1]).find(Boolean);
const elements = new Proxy({}, {
  get(target, key) {
    if (!target[key]) {
      target[key] = {
        value: '', checked: true, disabled: false, innerText: '',
        style: {}, classList: { add() {}, remove() {}, toggle() {} },
        addEventListener() {}, setAttribute() {}, removeAttribute() {},
        querySelectorAll() { return []; }
      };
    }
    return target[key];
  }
});
elements['current-report-number'].value = '  REPORT-A  ';
elements['pdf-page-select'].value = '1';
let fetchCalls = 0;
const context = {
  EnergyPdfRecognitionState: require('./static/energy/pdf_recognition_state.js'),
  document: {
    addEventListener() {}, getElementById(id) { return elements[id]; },
    querySelectorAll() { return []; }
  },
  window: { addEventListener() {}, scrollTo() {} },
  console, AbortController, FormData, alert() {},
  fetch: async () => {
    fetchCalls += 1;
    throw new Error('stale state attempted a request');
  },
  Chart: function Chart() {}
};
vm.createContext(context);
vm.runInContext(inlineScript, context);
vm.runInContext(`
  preparedPdf = { uploadToken: 'token-a' };
  pdfUploadSessionId = 1;
  const start = pdfRecognitionRequests.begin(
    currentPdfRecognitionSelection(), { abort() {} }
  );
  if (!pdfRecognitionRequests.accept(start.request, currentPdfRecognitionSelection())) {
    throw new Error('REPORT-A recognition was not accepted');
  }
  acceptedRecognitionReportNumber = 'REPORT-A';
  aiResultData = {
    report_number: 'REPORT-A',
    exterior_topology: { confirmed: true, load_geometry_ready: true }
  };
  if (!hasCurrentConfirmedGeometry()) throw new Error('REPORT-A was not ready');
  document.getElementById('current-report-number').value = 'REPORT-B';
  handleReportNumberInputChange();
  if (hasCurrentConfirmedGeometry()) throw new Error('stale REPORT-A stayed ready');
  if (aiResultData !== null || pendingExteriorFusion !== null) {
    throw new Error('report change did not clear derived state');
  }
  if (pdfRecognitionRequests.hasAccepted(currentPdfRecognitionSelection())) {
    throw new Error('REPORT-A fingerprint was accepted for REPORT-B');
  }
  calculateEnergy();
`, context);
if (fetchCalls !== 0) throw new Error('stale REPORT-A was sent as REPORT-B');
'''
        completed = subprocess.run(
            ["node", "-e", node_script],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_stale_energy_responses_cannot_land_after_report_identity_changes(self):
        node_script = r'''
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('templates/energy.html', 'utf8');
const inlineScript = [...html.matchAll(/<script(?:[^>]*)>([\s\S]*?)<\/script>/g)]
  .map(match => match[1]).find(Boolean);
const elements = new Proxy({}, {
  get(target, key) {
    if (!target[key]) {
      target[key] = {
        value: '', checked: true, disabled: false, innerText: '', innerHTML: '',
        style: {}, className: '',
        classList: { add() {}, remove() {}, toggle() {} },
        addEventListener() {}, setAttribute() {}, removeAttribute() {},
        querySelectorAll() { return []; }
      };
    }
    return target[key];
  }
});
elements['current-report-number'].value = 'REPORT-A';
elements['pdf-page-select'].value = '1';
elements['param-floors'].value = '1';
elements['param-window-repeat-count'].value = '';
const requests = [];
const tracking = { displays: [], steps: [], alerts: [], hideCount: 0 };
const context = {
  EnergyPdfRecognitionState: require('./static/energy/pdf_recognition_state.js'),
  document: {
    addEventListener() {}, getElementById(id) { return elements[id]; },
    querySelectorAll() { return []; }, createElement() { return elements.created; }
  },
  window: { addEventListener() {}, scrollTo() {} },
  console, AbortController, FormData,
  alert(message) { tracking.alerts.push(message); },
  fetch(url, options) {
    if (url !== '/energy/ai_simulate') throw new Error(`unexpected fetch ${url}`);
    let resolve;
    const response = new Promise(done => { resolve = done; });
    requests.push({
      reportNumber: JSON.parse(options.body).report_number,
      resolve
    });
    return response;
  },
  Chart: function Chart() {}, tracking
};

function acceptRecognition(reportNumber) {
  elements['current-report-number'].value = reportNumber;
  vm.runInContext(`
    {
      const accepted = pdfRecognitionRequests.begin(
        currentPdfRecognitionSelection(), { abort() {} }
      );
      if (!pdfRecognitionRequests.accept(accepted.request, currentPdfRecognitionSelection())) {
        throw new Error('could not accept recognition for ${reportNumber}');
      }
      pdfRecognitionRequests.finish(accepted.request, currentPdfRecognitionSelection());
    }
    acceptedRecognitionReportNumber = '${reportNumber}';
    reportNumber = '${reportNumber}';
    aiResultData = {
      report_number: '${reportNumber}',
      exterior_topology: { confirmed: true, load_geometry_ready: true }
    };
  `, context);
}

(async () => {
  vm.createContext(context);
  vm.runInContext(inlineScript, context);
  vm.runInContext(`
    displayResults = data => tracking.displays.push(data.marker);
    gotoStep = step => tracking.steps.push(step);
    showLoader = () => {};
    hideLoader = () => { tracking.hideCount += 1; };
  `, context);

  vm.runInContext(`preparedPdf = { uploadToken: 'token' }; pdfUploadSessionId = 1;`, context);
  acceptRecognition('REPORT-A');
  const calculationA = vm.runInContext('calculateEnergy()', context);

  elements['current-report-number'].value = 'REPORT-B';
  vm.runInContext('handleReportNumberInputChange()', context);
  if (tracking.hideCount !== 1) {
    throw new Error('report invalidation did not retire the abandoned REPORT-A loader');
  }
  acceptRecognition('REPORT-B');
  const calculationB = vm.runInContext('calculateEnergy()', context);

  if (requests.map(item => item.reportNumber).join(',') !== 'REPORT-A,REPORT-B') {
    throw new Error('calculation report numbers were not frozen per request');
  }
  requests[0].resolve({ json: async () => ({ marker: 'REPORT-A result' }) });
  await calculationA;
  if (vm.runInContext('energyResultData', context) !== null) {
    throw new Error('stale REPORT-A response wrote energyResultData');
  }
  if (tracking.displays.length || tracking.steps.length || tracking.hideCount !== 1) {
    throw new Error('stale REPORT-A response rendered, navigated, or hid REPORT-B loader: '
      + JSON.stringify(tracking));
  }

  requests[1].resolve({ json: async () => ({ marker: 'REPORT-B result' }) });
  await calculationB;
  if (vm.runInContext('energyResultData.marker', context) !== 'REPORT-B result') {
    throw new Error('current REPORT-B response did not land');
  }
  if (tracking.displays.join(',') !== 'REPORT-B result'
      || tracking.steps.join(',') !== '4' || tracking.hideCount !== 2) {
    throw new Error('current REPORT-B response did not render normally');
  }

  const staleError = vm.runInContext('calculateEnergy()', context);
  elements['current-report-number'].value = 'REPORT-C';
  vm.runInContext('handleReportNumberInputChange()', context);
  if (tracking.hideCount !== 3) {
    throw new Error('report invalidation did not retire the abandoned REPORT-B loader');
  }
  requests[2].resolve({ json: async () => ({ error: 'delayed REPORT-B failure' }) });
  await staleError;
  if (tracking.alerts.length || tracking.hideCount !== 3
      || tracking.displays.join(',') !== 'REPORT-B result'
      || tracking.steps.join(',') !== '4') {
    throw new Error('stale error/finally mutated the newer report state');
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
        completed = subprocess.run(
            ["node", "-e", node_script],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_pdf_region_recognition_ui_loads_preview_and_manages_crop_state(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        for function_name in [
            "loadPreparedPdfPreview",
            "resetPdfCropSelection",
            "setPdfRecognitionMode",
            "updatePdfCropOverlay",
        ]:
            with self.subTest(function_name=function_name):
                self.assertIn(f"function {function_name}", html)
        self.assertIn("fetch('/energy/pdf_page_preview'", html)

        prepare_body = re.search(
            r"async function preparePdf\(file\) \{(?P<body>.*?)\n        \}",
            html,
            re.S,
        ).group("body")
        self.assertNotIn("await recognizePreparedPdf()", prepare_body)
        self.assertIn("await loadPreparedPdfPreview()", prepare_body)

    def test_pdf_recognition_request_sends_mode_and_crop_coordinates(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn("fd.append('recognition_mode'", html)
        self.assertIn("fd.append('crop_bbox_px'", html)
        self.assertIn("fd.append('crop_preview_size'", html)

    def test_pdf_async_requests_guard_prepare_and_preview_generations(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn("let pdfUploadSessionId = 0;", html)
        self.assertIn("let pdfPreviewRequestId = 0;", html)
        self.assertIn("function advancePdfUploadSession()", html)
        self.assertIn("function advancePdfPreviewRequest()", html)
        self.assertIn("pdfPrepareAbortController.abort()", html)
        self.assertIn("pdfPreviewAbortController.abort()", html)

        prepare_segment = html[
            html.index("async function preparePdf(file")
            :html.index("async function recognizePreparedPdf()")
        ]
        self.assertIn(
            "!pdfPrepareSessionMatchesCurrent(uploadSessionId, pdfUploadSessionId)",
            prepare_segment,
        )
        self.assertLess(
            prepare_segment.index("!pdfPrepareSessionMatchesCurrent"),
            prepare_segment.index("preparedPdf = {"),
        )

        preview_segment = html[
            html.index("async function loadPreparedPdfPreview()")
            :html.index("function resetPreparedPdfSelection()")
        ]
        self.assertIn("const previewRequest = {", preview_segment)
        self.assertIn("signal: requestController.signal", preview_segment)
        self.assertIn("const decodedPreview = await decodePdfPreviewImage", preview_segment)
        decode_index = preview_segment.index(
            "const decodedPreview = await decodePdfPreviewImage"
        )
        post_decode_guard = preview_segment.index(
            "if (!isCurrentPdfPreviewRequest(previewRequest)) return false;",
            decode_index,
        )
        visible_commit = preview_segment.index(
            "previewImage.src = decodedPreview.src",
            post_decode_guard,
        )
        self.assertLess(decode_index, post_decode_guard)
        self.assertLess(post_decode_guard, visible_commit)

    def test_energy_request_and_server_forward_detailed_boundary_values(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        server = Path("web_server_server.py").read_text(encoding="utf-8")

        self.assertIn("include_roof: calculationMode === 'detailed'", html)
        self.assertIn("include_floor: calculationMode === 'detailed'", html)
        self.assertIn("annual_solar_irradiation_kwh_m2a:", html)
        self.assertIn('"include_roof": data.get("include_roof") is True', server)
        self.assertIn('"include_floor": data.get("include_floor") is True', server)
        self.assertIn('"annual_solar_irradiation_kwh_m2a": max(', server)

    def test_boundary_fields_are_disabled_only_in_detailed_mode(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn("function updateBoundaryControlState()", html)
        self.assertIn("const roofEnabled = !detailed ||", html)
        self.assertIn("const floorEnabled = !detailed ||", html)
        self.assertIn("updateBoundaryControlState();", html)

    def test_server_maps_and_validates_annual_use_fields(self):
        server = Path("web_server_server.py").read_text(encoding="utf-8")

        required_contract = [
            '"occupancy": {',
            '"area_per_person_m2": area_per_person',
            '"fresh_air_m3h_per_person": fresh_air_per_person',
            '"daily_operation_hours": daily_operation_hours',
            '"annual_operation_days": annual_operation_days',
            '"heat_recovery_efficiency": heat_recovery_percent / 100.0',
            '"infiltration_ach": infiltration_ach',
            '"fan_power_w_per_m3h": fan_power',
            '"seasonal_efficiency": heating_seasonal_efficiency',
            '"seasonal_efficiency": cooling_seasonal_efficiency',
            '"outdoor_design_temperature_c": winter_design_temperature',
            '"outdoor_design_temperature_c": summer_design_temperature',
        ]
        for fragment in required_contract:
            self.assertIn(fragment, server)

        for validation_fragment in [
            "def parse_finite_number",
            "math.isfinite(value)",
            "except (TypeError, ValueError)",
            "except ValueError as e:",
            '"heat_recovery_efficiency" in data',
            "heating_efficiency_defaults[heating_system]",
            "cooling_efficiency_defaults[cooling_system]",
        ]:
            self.assertIn(validation_fragment, server)

    def test_step_three_contains_global_annual_use_controls_and_defaults(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        expected = {
            "param-t-heat": "18.0",
            "param-t-cool": "26.0",
            "param-area-per-person": "10",
            "param-fresh-air-per-person": "30",
            "param-daily-operation-hours": "12",
            "param-annual-operation-days": "250",
            "param-heat-recovery-efficiency": "0",
            "param-infiltration-ach": "0",
            "param-fan-power": "0.5",
            "param-heating-seasonal-efficiency": "1.90",
            "param-cooling-seasonal-efficiency": "2.30",
            "param-winter-design-temperature": "-5",
            "param-summer-design-temperature": "34.9",
            "param-peak-solar-irradiance": "500",
            "param-summer-outdoor-rh": "60",
            "param-summer-indoor-rh": "50",
            "param-people-sensible": "75",
            "param-people-latent": "55",
            "param-stable-internal-gain-fraction": "0",
            "param-lpd": "8.0",
            "param-epd": "15.0",
        }
        for element_id, default_value in expected.items():
            input_tag = re.search(rf'<input[^>]*id="{element_id}"[^>]*>', html)
            self.assertIsNotNone(input_tag, element_id)
            self.assertIn(f'value="{default_value}"', input_tag.group(0))

        self.assertNotIn('id="param-room-type"', html)
        self.assertIn('id="annual-use-preview"', html)
        self.assertIn("function updateAnnualUsePreview()", html)
        self.assertIn('onchange="applyHeatingSystemEfficiencyDefault()"', html)
        self.assertIn('onchange="applyCoolingSystemEfficiencyDefault()"', html)
        self.assertIn("function applyHeatingSystemEfficiencyDefault()", html)
        self.assertIn("function applyCoolingSystemEfficiencyDefault()", html)
        self.assertIn("gas_boiler: 0.89", html)
        self.assertIn("coal_boiler: 0.75", html)
        self.assertIn("district: 0.80", html)

    def test_design_load_request_fields_are_forwarded_from_frontend_and_server(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        server = Path("web_server_server.py").read_text(encoding="utf-8")

        for fragment in [
            "peak_solar_irradiance_w_m2: parseFloat(document.getElementById('param-peak-solar-irradiance').value)",
            "summer_outdoor_relative_humidity_percent: parseFloat(document.getElementById('param-summer-outdoor-rh').value)",
            "summer_indoor_relative_humidity_percent: parseFloat(document.getElementById('param-summer-indoor-rh').value)",
            "sensible_heat_w_per_person: parseFloat(document.getElementById('param-people-sensible').value)",
            "latent_heat_w_per_person: parseFloat(document.getElementById('param-people-latent').value)",
            "stable_internal_gain_fraction: parseFloat(document.getElementById('param-stable-internal-gain-fraction').value)",
        ]:
            self.assertIn(fragment, html)

        for fragment in [
            'peak_solar_irradiance = parse_finite_number(',
            '"peak_solar_irradiance_w_m2", 500.0',
            'people_sensible = parse_finite_number("sensible_heat_w_per_person"',
            'people_latent = parse_finite_number("latent_heat_w_per_person"',
            'summer_outdoor_rh = parse_finite_number(',
            '"summer_outdoor_relative_humidity_percent", 60.0',
            'summer_indoor_rh = parse_finite_number(',
            '"summer_indoor_relative_humidity_percent", 50.0',
            'stable_internal_gain_fraction = parse_finite_number(',
            '"stable_internal_gain_fraction", 0.0',
            '"peak_solar_irradiance_w_m2": peak_solar_irradiance',
            '"sensible_heat_w_per_person": people_sensible',
            '"latent_heat_w_per_person": people_latent',
            '"summer_outdoor_relative_humidity_percent": summer_outdoor_rh',
            '"summer_indoor_relative_humidity_percent": summer_indoor_rh',
            '"stable_internal_gain_fraction": stable_internal_gain_fraction',
        ]:
            self.assertIn(fragment, server)

    def test_other_design_parameters_are_configurable_in_collapsed_panel(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        server = Path("web_server_server.py").read_text(encoding="utf-8")

        for fragment in [
            'onclick="toggleOtherDesignParams()"',
            'id="other-design-params-panel"',
            '其他参数',
            'id="param-door-invasion-heat"',
            'id="param-heating-addition-factor"',
            'id="param-cooling-load-factor"',
            'id="param-lighting-cooling-load-factor"',
            'id="param-equipment-cooling-load-factor"',
            'id="param-air-density"',
            "function toggleOtherDesignParams()",
            "door_invasion_heat_w: parseFloat(document.getElementById('param-door-invasion-heat').value)",
            "heating_addition_factor: parseFloat(document.getElementById('param-heating-addition-factor').value)",
            "cooling_load_factor: parseFloat(document.getElementById('param-cooling-load-factor').value)",
            "lighting_cooling_load_factor: parseFloat(document.getElementById('param-lighting-cooling-load-factor').value)",
            "equipment_cooling_load_factor: parseFloat(document.getElementById('param-equipment-cooling-load-factor').value)",
            "air_density_kg_m3: parseFloat(document.getElementById('param-air-density').value)",
            "heating.door_invasion_heat_w ?? p.door_invasion_heat_w ?? 0",
            "heating.heating_addition_factor ?? p.heating_addition_factor ?? 1",
            "occupancy.cooling_load_factor ?? p.cooling_load_factor ?? 1",
            "lighting.cooling_load_factor ?? p.lighting_cooling_load_factor ?? 1",
            "equipment.cooling_load_factor ?? p.equipment_cooling_load_factor ?? 1",
            "ventilation.air_density_kg_m3 ?? p.air_density_kg_m3 ?? 1.13",
        ]:
            self.assertIn(fragment, html)

        for fragment in [
            'door_invasion_heat = parse_finite_number("door_invasion_heat_w"',
            'heating_addition_factor = parse_finite_number("heating_addition_factor"',
            'cooling_load_factor = parse_finite_number("cooling_load_factor"',
            'lighting_cooling_load_factor = parse_finite_number("lighting_cooling_load_factor"',
            'equipment_cooling_load_factor = parse_finite_number("equipment_cooling_load_factor"',
            'air_density = parse_finite_number("air_density_kg_m3"',
            '"door_invasion_heat_w": door_invasion_heat',
            '"heating_addition_factor": heating_addition_factor',
            '"cooling_load_factor": cooling_load_factor',
            '"cooling_load_factor": lighting_cooling_load_factor',
            '"cooling_load_factor": equipment_cooling_load_factor',
            '"air_density_kg_m3": air_density',
        ]:
            self.assertIn(fragment, server)

    def test_step_three_has_independent_heating_and_cooling_scope_controls(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        heating = re.search(r'<input[^>]*id="calculate-heating"[^>]*>', html)
        cooling = re.search(r'<input[^>]*id="calculate-cooling"[^>]*>', html)
        self.assertIsNotNone(heating)
        self.assertIsNotNone(cooling)
        self.assertIn("checked", heating.group(0))
        self.assertIn("checked", cooling.group(0))
        self.assertIn('id="heating-calculation-controls"', html)
        self.assertIn('id="cooling-calculation-controls"', html)
        self.assertIn("function updateCalculationScopeControls()", html)
        self.assertIn("control.disabled = !heatingEnabled", html)
        self.assertIn("control.disabled = !coolingEnabled", html)

        calculate_segment = html[html.index("async function calculateEnergy()"):html.index("function displayResults(data)")]
        self.assertIn("const calculateHeating", calculate_segment)
        self.assertIn("const calculateCooling", calculate_segment)
        self.assertIn("if (!calculateHeating && !calculateCooling)", calculate_segment)
        self.assertIn("calculate_heating: calculateHeating", calculate_segment)
        self.assertIn("calculate_cooling: calculateCooling", calculate_segment)

    def test_history_restores_new_and_legacy_calculation_scope(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        history_segment = html[html.index("async function loadReportDetails(repNum)"):html.index("function showLoader(text)")]

        self.assertIn("function inferCalculationEnabled", html)
        self.assertIn("typeof section.enabled === 'boolean'", html)
        self.assertIn("systemType !== 'none'", html)
        self.assertIn("document.getElementById('calculate-heating').checked", history_segment)
        self.assertIn("document.getElementById('calculate-cooling').checked", history_segment)
        self.assertIn("updateCalculationScopeControls();", history_segment)

    def test_results_mark_disabled_calculation_scope_as_not_enabled(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        display_segment = html[html.index("function displayResults(data)"):html.index("async function loadReportDetails(repNum)")]

        self.assertIn("data.calculation_scope", display_segment)
        self.assertIn("未启用（本报告不计算", display_segment)
        self.assertIn("scope.heating", display_segment)
        self.assertIn("scope.cooling", display_segment)
        self.assertIn("（未启用）", display_segment)

    def test_results_separate_thermal_demand_from_electricity(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn('id="thermal-demand-results"', html)
        self.assertIn('id="electricity-results"', html)
        self.assertIn("annual_thermal_demand_kwh_th", html)
        self.assertIn("annual_electricity_kwh", html)
        self.assertIn("equivalent_peak_check_kw_th", html)
        self.assertIn("kWhₜₕ/a", html)
        self.assertIn("kWh/a", html)
        self.assertIn("等效峰值校核", html)
        self.assertIn("model_boundaries", html)
        self.assertIn('id="non-electric-purchased-results"', html)
        self.assertIn("annual_non_electric_purchased_energy_kwh", html)
        self.assertIn("非电购入能源", html)
        self.assertIn("const hasModernElectricity = Boolean(data.annual_electricity_kwh)", html)
        self.assertNotIn("legacyElectricity[cat.legacyKey]", html)

    def test_results_render_design_heating_and_cooling_loads(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        for fragment in [
            'id="design-load-results"',
            'id="design-heating-total"',
            'id="design-cooling-total"',
            'id="design-heating-index"',
            'id="design-cooling-index"',
            'id="design-load-tbody"',
            "data.design_loads || {}",
            "renderDesignLoadResults",
            "heating_design_load",
            "cooling_design_load",
        ]:
            self.assertIn(fragment, html)

    def test_step_four_navigation_uses_energy_result_state(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        goto_segment = html[html.index("function gotoStep(step)"):html.index("// 上传与识别逻辑")]
        calculate_segment = html[html.index("async function calculateEnergy()"):html.index("function displayResults(data)")]
        history_segment = html[html.index("async function loadReportDetails(repNum)"):html.index("function showLoader(text)")]

        self.assertIn("let energyResultData = null;", html)
        self.assertIn("step === 4 && !energyResultData", goto_segment)
        self.assertNotIn("step === 4 && !aiResultData", goto_segment)

        calculated_state = calculate_segment.index("energyResultData = data;")
        calculated_render = calculate_segment.index("displayResults(energyResultData);")
        calculated_navigation = calculate_segment.index("gotoStep(4);")
        self.assertLess(calculated_state, calculated_render)
        self.assertLess(calculated_render, calculated_navigation)

        history_state = history_segment.index("energyResultData = data.results;")
        history_render = history_segment.index("displayResults(energyResultData);")
        history_navigation = history_segment.index("gotoStep(4);")
        self.assertLess(history_state, history_render)
        self.assertLess(history_render, history_navigation)

    def test_energy_result_state_is_invalidated_for_new_work(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")
        recognize_segment = html[html.index("async function handleFile(file)"):html.index("function updateAnnualUsePreview()")]
        calculate_segment = html[html.index("async function calculateEnergy()"):html.index("function displayResults(data)")]
        calculate_failure = calculate_segment[calculate_segment.index("} catch (error) {"):]

        self.assertIn('onclick="resetEnergyAssessment()"', html)
        self.assertIn("function invalidateEnergyResult()", html)
        self.assertIn("energyResultData = null;", html[html.index("function invalidateEnergyResult()"):html.index("function resetEnergyAssessment()")])
        reset_segment = html[html.index("function resetEnergyAssessment()"):html.index("async function handleFile(file)")]
        self.assertLess(reset_segment.index("invalidateEnergyResult();"), reset_segment.index("gotoStep(1);"))

        self.assertLess(
            recognize_segment.index("invalidatePdfRecognitionSelection("),
            recognize_segment.index("fetch(recognitionEndpoint"),
        )
        self.assertLess(calculate_segment.index("invalidateEnergyResult();"), calculate_segment.index("fetch('/energy/ai_simulate'"))
        self.assertNotIn("energyResultData =", calculate_failure)

    def test_non_xian_city_warns_about_xian_design_temperature_defaults(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        self.assertIn('id="design-temperature-warning"', html)
        self.assertIn("function updateDesignTemperatureNotice()", html)
        self.assertIn("当前峰值校核温度为西安参考值", html)
        self.assertIn("updateDesignTemperatureNotice()", html)

    def test_history_reload_supports_nested_and_flat_annual_use_fields(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        for fragment in [
            "occupancy.area_per_person_m2 ?? p.area_per_person_m2 ?? 10",
            "ventilation.fresh_air_m3h_per_person ?? p.fresh_air_m3h_per_person ?? 30",
            "schedule.daily_operation_hours ?? p.daily_operation_hours ??",
            "schedule.annual_operation_days ?? p.annual_operation_days ?? 250",
            "p.heat_recovery_efficiency_percent ?? 0",
            "p.heat_recovery_efficiency ??",
            "p.op_hours",
            "heating.seasonal_efficiency ?? p.heating_seasonal_efficiency ??",
            "cooling.seasonal_efficiency ?? p.cooling_seasonal_efficiency ??",
            "data.breakdown_kwh || {}",
            "updateAnnualUsePreview();",
        ]:
            self.assertIn(fragment, html)

        self.assertIn("setCalculationMode(p.calculation_mode || 'simple')", html)
        self.assertIn("detailedEnvelope = normalizeDetailedEnvelope(p.detailed_envelope)", html)

    def test_detailed_history_normalizes_empty_and_partial_shapes(self):
        html = Path("templates/energy.html").read_text(encoding="utf-8")

        for fragment in [
            "function createDefaultDetailedEnvelope()",
            "function normalizeDetailedEnvelope(savedEnvelope)",
            "walls_by_orientation: {}",
            "windows_by_orientation: {}",
            "roof: {",
            "ground: {",
            "...defaults.walls_by_orientation[key]",
            "...objectOrEmpty(source.walls_by_orientation?.[key])",
            "...defaults.windows_by_orientation[key]",
            "...objectOrEmpty(source.windows_by_orientation?.[key])",
            "roof: { ...defaults.roof, ...objectOrEmpty(source.roof) }",
            "ground: { ...defaults.ground, ...objectOrEmpty(source.ground) }",
            "detailedEnvelope = normalizeDetailedEnvelope(p.detailed_envelope)",
        ]:
            self.assertIn(fragment, html)

        self.assertNotIn("detailedEnvelope = p.detailed_envelope ||", html)


class EnergyRouteClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="test-secret")

    def setUp(self):
        self.client = self.server.app.test_client()
        with self.client.session_transaction() as sess:
            sess["logged_in"] = True
            sess["username"] = "route-test"

    def test_partial_closure_never_enables_load_geometry(self):
        topology = {"room_count": 5, "load_geometry_ready": True}

        result = self.server._enforce_topology_repair_readiness(
            topology,
            {"closure_status": "partial", "manual_exterior_wall_required": False},
        )

        self.assertFalse(result["load_geometry_ready"])

    @staticmethod
    def make_pdf_bytes(page_labels):
        from reportlab.pdfgen import canvas

        pdf_bytes = io.BytesIO()
        pdf = canvas.Canvas(pdf_bytes, pagesize=(400, 300))
        for label in page_labels:
            pdf.drawString(40, 260, label)
            pdf.showPage()
        pdf.save()
        pdf_bytes.seek(0)
        return pdf_bytes

    @staticmethod
    def make_pdf_segmenter(result):
        from floorplan_onnx import FloorplanSegmenterONNX

        real_preprocessor = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        real_preprocessor.img_size = 512
        segmenter = MagicMock()
        segmenter.remove_annotations.side_effect = real_preprocessor.remove_annotations
        segmenter.preprocess_dark_cad.side_effect = real_preprocessor.preprocess_dark_cad
        segmenter.prepare_model_input.side_effect = real_preprocessor.prepare_model_input
        segmenter.predict.return_value = result
        return segmenter

    def test_pdf_prepare_saves_upload_once_and_returns_page_count(self):
        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                response = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "MULTIPAGE",
                        "raster_file": (self.make_pdf_bytes(["PAGE-1", "PAGE-2"]), "floors.pdf"),
                    },
                    content_type="multipart/form-data",
                )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            body = response.get_json()
            self.assertTrue(body["success"])
            self.assertEqual(body["page_count"], 2)
            self.assertEqual(body["filename"], "floors.pdf")
            self.assertTrue(body["upload_token"])
            prepared_files = list((Path(upload_root) / "energy" / "MULTIPAGE").glob("building_plan_prepared_*.pdf"))
            self.assertEqual(len(prepared_files), 1)

    def test_pdf_prepare_rejects_non_pdf_and_unreadable_pdf(self):
        for file_bytes, filename in [(b"not pdf", "plan.png"), (b"not pdf", "plan.pdf")]:
            with self.subTest(filename=filename):
                response = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "BAD-PDF",
                        "raster_file": (io.BytesIO(file_bytes), filename),
                    },
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_resolve_poppler_path_finds_bundled_windows_binaries(self):
        with tempfile.TemporaryDirectory() as directory:
            poppler_bin = Path(directory) / "poppler" / "Library" / "bin"
            poppler_bin.mkdir(parents=True)
            (poppler_bin / "pdfinfo.exe").touch()
            (poppler_bin / "pdftoppm.exe").touch()

            with patch.dict(os.environ, {"POPPLER_PATH": ""}):
                resolved = self.server._resolve_poppler_path([poppler_bin])

            self.assertEqual(resolved, str(poppler_bin))

    def test_pdf_page_preview_returns_selected_page_as_jpeg(self):
        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                prepared = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "REGION-PREVIEW",
                        "raster_file": (
                            self.make_pdf_bytes(["PAGE-1", "PAGE-2"]),
                            "floors.pdf",
                        ),
                    },
                    content_type="multipart/form-data",
                ).get_json()
                preview_bgr = np.full((600, 800, 3), 128, dtype=np.uint8)
                with (
                    patch.object(
                        self.server,
                        "render_pdf_page_preview",
                        return_value=preview_bgr,
                        create=True,
                    ) as render_preview,
                    patch.object(
                        self.server,
                        "_resolve_poppler_path",
                        return_value="C:\\poppler\\bin",
                    ),
                ):
                    response = self.client.post(
                        "/energy/pdf_page_preview",
                        data={
                            "report_number": "REGION-PREVIEW",
                            "pdf_upload_token": prepared["upload_token"],
                            "pdf_page_number": "2",
                        },
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            body = response.get_json()
            self.assertTrue(body["success"])
            self.assertEqual(body["pdf_page_number"], 2)
            self.assertEqual(body["pdf_page_count"], 2)
            self.assertEqual(body["image_size"], [800, 600])
            jpeg = cv2.imdecode(
                np.frombuffer(base64.b64decode(body["image"]), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertIsNotNone(jpeg)
            self.assertEqual(jpeg.shape[:2], (600, 800))
            stored_pdf = next(
                (Path(upload_root) / "energy" / "REGION-PREVIEW").glob(
                    "building_plan_prepared_*.pdf"
                )
            )
            render_preview.assert_called_once_with(
                stored_pdf,
                2,
                2,
                "C:\\poppler\\bin",
            )

    def test_pdf_page_preview_rejects_mismatched_report_or_invalid_page(self):
        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                prepared = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "REGION-PREVIEW",
                        "raster_file": (
                            self.make_pdf_bytes(["PAGE-1", "PAGE-2"]),
                            "floors.pdf",
                        ),
                    },
                    content_type="multipart/form-data",
                ).get_json()
                for report_number, page_number, error_fragment in [
                    ("OTHER-REPORT", "1", "report"),
                    ("REGION-PREVIEW", "3", "page"),
                ]:
                    with self.subTest(report_number=report_number, page_number=page_number):
                        response = self.client.post(
                            "/energy/pdf_page_preview",
                            data={
                                "report_number": report_number,
                                "pdf_upload_token": prepared["upload_token"],
                                "pdf_page_number": page_number,
                            },
                        )
                        self.assertEqual(response.status_code, 400)
                        self.assertIn(error_fragment, response.get_json()["error"].lower())
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

    def test_crop_region_helper_route_serves_shared_source(self):
        response = self.client.get("/energy/crop_region.js")
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        self.assertIn("javascript", response.mimetype)
        self.assertEqual(
            response.get_data(),
            Path("static/energy/crop_region.js").read_bytes(),
        )

    def test_pdf_recognition_state_helper_is_served_to_the_energy_ui(self):
        response = self.client.get("/energy/pdf_recognition_state.js")
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        self.assertIn("javascript", response.mimetype)
        self.assertEqual(
            response.get_data(),
            Path("static/energy/pdf_recognition_state.js").read_bytes(),
        )

    def test_energy_assets_do_not_depend_on_annotation_tool_directory(self):
        source = Path("web_server_server.py").read_text(encoding="utf-8")
        self.assertNotIn("annotation_tool', 'static", source)

    def test_ai_recognize_rejects_invalid_crop_region_requests(self):
        from floorplan_page_pipeline import PreparedFloorplanPage

        prediction = {
            "mask": np.zeros((2, 2), dtype=np.uint8),
            "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
            "stats": {},
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {
                "status": "no_closed_rooms",
                "room_count": 0,
                "rooms": [],
                "total_area_px2": 0.0,
                "total_area_m2": None,
                "load_geometry_ready": False,
            },
            "image_size": [2, 2],
        }
        prepared_page = PreparedFloorplanPage(
            page_number=1,
            page_count=1,
            render_bgr=np.full((2, 2, 3), 255, dtype=np.uint8),
            cleaned_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            model_view_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            model_input_512=np.zeros((1, 3, 512, 512), dtype=np.float32),
            model_input_metadata={},
            vector_analysis={},
            scale_calibration={"status": "manual_required", "method": "test"},
            vector_cleanup={
                "enabled": False,
                "has_vector_geometry": False,
                "has_vector_text": False,
            },
            inference_roi=None,
            cleanup_mask=np.zeros((2, 2), dtype=np.uint8),
            structural_support_mask=None,
            artifacts={},
        )

        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            report_dir = Path(upload_root) / "energy" / "CROP-INVALID"
            report_dir.mkdir(parents=True)
            stored_filename = "building_plan_prepared_test.pdf"
            (report_dir / stored_filename).write_bytes(b"%PDF-1.4")
            upload_token = self.server._make_pdf_upload_token(
                "CROP-INVALID",
                stored_filename,
                1,
            )
            segmenter = MagicMock()
            segmenter.predict.return_value = prediction
            huge_pixel_value = 10 ** 1000
            cases = [
                (
                    "crop mode with a raster upload",
                    {
                        "report_number": "CROP-INVALID",
                        "recognition_mode": "crop_region",
                        "crop_bbox_px": "[100, 50, 500, 350]",
                        "crop_preview_size": "[800, 600]",
                        "raster_file": (io.BytesIO(b"fake image"), "plan.png"),
                    },
                ),
                (
                    "malformed crop JSON",
                    {
                        "report_number": "CROP-INVALID",
                        "recognition_mode": "crop_region",
                        "crop_bbox_px": "[100, 50",
                        "crop_preview_size": "[800, 600]",
                        "pdf_upload_token": upload_token,
                        "pdf_page_number": "1",
                    },
                ),
                (
                    "crop outside declared preview bounds",
                    {
                        "report_number": "CROP-INVALID",
                        "recognition_mode": "crop_region",
                        "crop_bbox_px": "[100, 50, 801, 350]",
                        "crop_preview_size": "[800, 600]",
                        "pdf_upload_token": upload_token,
                        "pdf_page_number": "1",
                    },
                ),
                (
                    "crop width below minimum dimensions",
                    {
                        "report_number": "CROP-INVALID",
                        "recognition_mode": "crop_region",
                        "crop_bbox_px": "[100, 50, 227, 350]",
                        "crop_preview_size": "[800, 600]",
                        "pdf_upload_token": upload_token,
                        "pdf_page_number": "1",
                    },
                ),
                (
                    "crop height below minimum dimensions",
                    {
                        "report_number": "CROP-INVALID",
                        "recognition_mode": "crop_region",
                        "crop_bbox_px": "[100, 50, 500, 177]",
                        "crop_preview_size": "[800, 600]",
                        "pdf_upload_token": upload_token,
                        "pdf_page_number": "1",
                    },
                ),
                (
                    "crop area below one percent",
                    {
                        "report_number": "CROP-INVALID",
                        "recognition_mode": "crop_region",
                        "crop_bbox_px": "[100, 50, 228, 178]",
                        "crop_preview_size": "[2000, 1000]",
                        "pdf_upload_token": upload_token,
                        "pdf_page_number": "1",
                    },
                ),
                (
                    "unreasonably large crop integer",
                    {
                        "report_number": "CROP-INVALID",
                        "recognition_mode": "crop_region",
                        "crop_bbox_px": f"[0, 0, 128, {huge_pixel_value}]",
                        "crop_preview_size": f"[128, {huge_pixel_value}]",
                        "pdf_upload_token": upload_token,
                        "pdf_page_number": "1",
                    },
                ),
            ]
            try:
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(
                        self.server,
                        "_floorplan_segmenter",
                        segmenter,
                        create=True,
                    ),
                    patch.object(
                        self.server,
                        "prepare_pdf_page",
                        return_value=prepared_page,
                        create=True,
                    ) as prepare_page,
                    patch.object(
                        self.server.cv2,
                        "imread",
                        return_value=np.zeros((2, 2, 3), dtype=np.uint8),
                    ),
                ):
                    for label, data in cases:
                        with self.subTest(case=label):
                            response = self.client.post(
                                "/energy/ai_recognize",
                                data=data,
                                content_type="multipart/form-data",
                            )
                            self.assertEqual(
                                response.status_code,
                                400,
                                response.get_json(),
                            )
                    prepare_page.assert_not_called()
                    segmenter.predict.assert_not_called()
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

    def test_ai_recognize_legacy_pdf_request_defaults_to_full_page(self):
        from floorplan_page_pipeline import PreparedFloorplanPage

        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            report_dir = Path(upload_root) / "energy" / "LEGACY-PDF"
            report_dir.mkdir(parents=True)
            stored_filename = "building_plan_prepared_test.pdf"
            (report_dir / stored_filename).write_bytes(b"%PDF-1.4")
            upload_token = self.server._make_pdf_upload_token(
                "LEGACY-PDF",
                stored_filename,
                1,
            )
            prepared_page = PreparedFloorplanPage(
                page_number=1,
                page_count=1,
                render_bgr=np.full((2, 2, 3), 255, dtype=np.uint8),
                cleaned_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
                model_view_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                model_input_512=np.zeros((1, 3, 512, 512), dtype=np.float32),
                model_input_metadata={},
                vector_analysis={},
                scale_calibration={"status": "manual_required", "method": "test"},
                vector_cleanup={
                    "enabled": False,
                    "has_vector_geometry": False,
                    "has_vector_text": False,
                },
                inference_roi=None,
                cleanup_mask=np.zeros((2, 2), dtype=np.uint8),
                structural_support_mask=None,
                artifacts={},
            )
            segmenter = MagicMock()
            segmenter.predict.return_value = {
                "mask": np.zeros((2, 2), dtype=np.uint8),
                "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
                "stats": {},
                "geometry": {"walls": [], "windows": [], "doors": []},
                "room_topology": {
                    "status": "no_closed_rooms",
                    "room_count": 0,
                    "rooms": [],
                    "total_area_px2": 0.0,
                    "total_area_m2": None,
                    "load_geometry_ready": False,
                },
                "image_size": [2, 2],
            }
            try:
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(
                        self.server,
                        "_floorplan_segmenter",
                        segmenter,
                        create=True,
                    ),
                    patch.object(
                        self.server,
                        "prepare_pdf_page",
                        return_value=prepared_page,
                        create=True,
                    ),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "LEGACY-PDF",
                            "pdf_upload_token": upload_token,
                            "pdf_page_number": "1",
                        },
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            body = response.get_json()
            saved = json.loads(
                (report_dir / "recognition.json").read_text(encoding="utf-8")
            )
            expected_region_metadata = {
                "recognition_mode": "full_page",
                "crop_bbox_page_px": None,
                "crop_bbox_preview_px": None,
                "crop_preview_size": None,
            }
            for field, expected in expected_region_metadata.items():
                with self.subTest(location="response", field=field):
                    self.assertIn(field, body)
                    self.assertEqual(body[field], expected)
                with self.subTest(location="recognition.json", field=field):
                    self.assertIn(field, saved)
                    self.assertEqual(saved[field], expected)

    def test_ai_recognize_crops_aligned_prepared_page_inputs_for_inference(self):
        from floorplan_page_pipeline import PreparedFloorplanPage

        page_height, page_width = 1200, 1600
        scale_calibration = {
            "status": "manual_required",
            "method": "whole_page_test",
            "scale_m_per_px": None,
            "evidence": [{"text": "40600", "source": "whole_page"}],
        }
        vector_cleanup = {
            "enabled": True,
            "has_vector_geometry": True,
            "has_vector_text": True,
            "structural_mask": {"enabled": True},
            "nonstructural_mask": {"enabled": True},
            "building_roi": {
                "enabled": True,
                "bbox_px": [250, 150, 900, 650],
                "confidence": 0.95,
                "reason": "whole_page_test",
            },
            "evidence": [{"kind": "whole_page_vector"}],
        }
        rows, columns = np.indices((page_height, page_width))
        render_bgr = np.stack(
            (
                columns % 251,
                rows % 251,
                (rows + columns) % 251,
            ),
            axis=2,
        ).astype(np.uint8)
        cleaned_bgr = (255 - render_bgr).astype(np.uint8)
        structural_support_mask = ((rows + columns) % 256).astype(np.uint8)
        prepared_page = PreparedFloorplanPage(
            page_number=1,
            page_count=1,
            render_bgr=render_bgr,
            cleaned_bgr=cleaned_bgr,
            model_view_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            model_input_512=np.zeros((1, 3, 512, 512), dtype=np.float32),
            model_input_metadata={},
            vector_analysis={"status": "completed"},
            scale_calibration=scale_calibration,
            vector_cleanup=vector_cleanup,
            inference_roi=[250, 150, 900, 650],
            cleanup_mask=np.zeros((page_height, page_width), dtype=np.uint8),
            structural_support_mask=structural_support_mask,
            artifacts={},
        )

        def predict(image, **kwargs):
            height, width = image.shape[:2]
            return {
                "mask": np.zeros((height, width), dtype=np.uint8),
                "overlay": np.zeros((height, width, 3), dtype=np.uint8),
                "stats": {},
                "geometry": {
                    "walls": [{
                        "pts": [[20, 30], [780, 30]],
                        "area": 760.0,
                        "bbox": [20, 30, 761, 1],
                    }],
                    "windows": [],
                    "doors": [],
                },
                "room_topology": {
                    "status": "closed",
                    "room_count": 1,
                    "rooms": [{
                        "id": "room-crop",
                        "polygon_px": [[40, 50], [760, 50], [760, 550], [40, 550]],
                        "area_px2": 360000.0,
                        "area_m2": None,
                    }],
                    "total_area_px2": 360000.0,
                    "total_area_m2": None,
                    "load_geometry_ready": False,
                },
                "image_size": [width, height],
            }

        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            report_dir = Path(upload_root) / "energy" / "CROP-INTEGRATION"
            report_dir.mkdir(parents=True)
            stale_mask_path = report_dir / "pdf_nonstructural_mask.png"
            self.assertTrue(
                cv2.imwrite(
                    str(stale_mask_path),
                    np.full((7, 9), 255, dtype=np.uint8),
                )
            )
            stored_filename = "building_plan_prepared_test.pdf"
            (report_dir / stored_filename).write_bytes(b"%PDF-1.4")
            upload_token = self.server._make_pdf_upload_token(
                "CROP-INTEGRATION",
                stored_filename,
                1,
            )
            segmenter = MagicMock()
            segmenter.predict.side_effect = predict
            try:
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(
                        self.server,
                        "_floorplan_segmenter",
                        segmenter,
                        create=True,
                    ),
                    patch.object(
                        self.server,
                        "prepare_pdf_page",
                        return_value=prepared_page,
                        create=True,
                    ),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "CROP-INTEGRATION",
                            "recognition_mode": "crop_region",
                            "crop_bbox_px": "[100, 50, 500, 350]",
                            "crop_preview_size": "[800, 600]",
                            "pdf_upload_token": upload_token,
                            "pdf_page_number": "1",
                        },
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            predicted_image = segmenter.predict.call_args.args[0]
            predict_kwargs = segmenter.predict.call_args.kwargs
            self.assertEqual(predicted_image.shape[:2], (600, 800))
            self.assertEqual(
                predicted_image[0, 0].tolist(),
                cleaned_bgr[100, 200].tolist(),
            )
            self.assertEqual(predict_kwargs["inference_roi"], [50, 50, 700, 550])
            repair_context = predict_kwargs["topology_repair_context"]
            self.assertEqual(
                repair_context["structural_support_mask"].shape,
                (600, 800),
            )
            self.assertEqual(
                int(repair_context["structural_support_mask"][0, 0]),
                int(structural_support_mask[100, 200]),
            )
            self.assertEqual(
                repair_context["building_roi"],
                [50, 50, 700, 550],
            )

            body = response.get_json()
            self.assertEqual(body["image_size"], [800, 600])
            self.assertEqual(body["scale_calibration"], scale_calibration)
            self.assertEqual(
                body["vector_cleanup"]["building_roi"]["bbox_px"],
                [50, 50, 700, 550],
            )
            self.assertEqual(
                body["vector_cleanup"]["evidence"],
                [{"kind": "whole_page_vector"}],
            )
            original = cv2.imdecode(
                np.frombuffer(
                    base64.b64decode(body["images"]["original"]),
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )
            self.assertEqual(original.shape[:2], (600, 800))
            overlay = cv2.imdecode(
                np.frombuffer(
                    base64.b64decode(body["images"]["overlay"]),
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )
            self.assertIsNotNone(overlay)
            self.assertEqual(overlay.shape[:2], (600, 800))
            self.assertTrue(body["geometry"]["walls"])
            for x, y in body["geometry"]["walls"][0]["pts"]:
                self.assertGreaterEqual(x, 0)
                self.assertLessEqual(x, 800)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(y, 600)
            self.assertTrue(body["room_topology"]["rooms"])
            for x, y in body["room_topology"]["rooms"][0]["polygon_px"]:
                self.assertGreaterEqual(x, 0)
                self.assertLessEqual(x, 800)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(y, 600)
            persisted_original = cv2.imread(
                str(report_dir / "building_plan_ai.png"),
            )
            self.assertEqual(persisted_original.shape[:2], (600, 800))
            self.assertEqual(
                persisted_original[0, 0].tolist(),
                render_bgr[100, 200].tolist(),
            )
            saved = json.loads(
                (report_dir / "recognition.json").read_text(encoding="utf-8")
            )
            saved_nonstructural_mask = cv2.imread(
                str(stale_mask_path),
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertIsNotNone(saved_nonstructural_mask)
            self.assertEqual(saved_nonstructural_mask.shape, (600, 800))
            self.assertEqual(int(np.count_nonzero(saved_nonstructural_mask)), 0)
            self.assertEqual(
                saved["artifacts"]["pdf_nonstructural_mask"],
                "pdf_nonstructural_mask.png",
            )
            self.assertTrue(saved["geometry"]["walls"])
            self.assertTrue(saved["room_topology"]["rooms"])
            expected_region_metadata = {
                "recognition_mode": "crop_region",
                "crop_bbox_page_px": [200, 100, 1000, 700],
                "crop_bbox_preview_px": [100, 50, 500, 350],
                "crop_preview_size": [800, 600],
            }
            for field, expected in expected_region_metadata.items():
                with self.subTest(location="response", field=field):
                    self.assertIn(field, body)
                    self.assertEqual(body[field], expected)
                with self.subTest(location="recognition.json", field=field):
                    self.assertIn(field, saved)
                    self.assertEqual(saved[field], expected)

    def test_ai_recognize_uses_prepared_pdf_selected_page_and_persists_metadata(self):
        from PIL import Image
        from floorplan_page_pipeline import VectorAnalysisOutcome

        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                prepared = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "PAGE-TWO",
                        "raster_file": (self.make_pdf_bytes(["PAGE-1", "PAGE-2"]), "floors.pdf"),
                    },
                    content_type="multipart/form-data",
                ).get_json()
                segmenter = MagicMock()
                from floorplan_onnx import FloorplanSegmenterONNX
                real_preprocessor = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
                real_preprocessor.img_size = 512
                segmenter.remove_annotations.side_effect = real_preprocessor.remove_annotations
                segmenter.preprocess_dark_cad.side_effect = real_preprocessor.preprocess_dark_cad
                segmenter.prepare_model_input.side_effect = real_preprocessor.prepare_model_input
                segmenter.predict.return_value = {
                    "mask": np.zeros((2, 2), dtype=np.uint8),
                    "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
                    "stats": {},
                    "geometry": {"walls": [], "windows": [], "doors": []},
                    "room_topology": {
                        "status": "no_closed_rooms", "room_count": 0, "rooms": [],
                        "total_area_px2": 0.0, "total_area_m2": None,
                        "scale_m_per_px": None, "load_geometry_ready": False,
                    },
                    "image_size": [2, 2],
                }
                page_data = {"is_vector_pdf": False, "page_index": 1, "page_count": 2}
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(self.server, "HAS_VECTOR_PDF_SCALE", True),
                    patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
                    patch(
                        "floorplan_page_pipeline._run_vector_analysis",
                        return_value=VectorAnalysisOutcome(
                            page_data,
                            {
                                "status": "completed",
                                "mode": "vector",
                                "timeout_seconds": 15.0,
                                "reason": None,
                            },
                        ),
                    ) as run_analysis,
                    patch(
                        "floorplan_page_pipeline.convert_from_path",
                        return_value=[Image.new("RGB", (2, 2), "white")],
                    ) as render_page,
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "PAGE-TWO",
                            "preprocessing": "auto",
                            "pdf_upload_token": prepared["upload_token"],
                            "pdf_page_number": "2",
                        },
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            run_analysis.assert_called_once_with(
                next((Path(upload_root) / "energy" / "PAGE-TWO").glob("building_plan_prepared_*.pdf")).resolve(),
                page_index=1,
                dpi=100,
                timeout_seconds=30.0,
            )
            self.assertEqual(render_page.call_args.kwargs["dpi"], 200)
            self.assertEqual(render_page.call_args.kwargs["first_page"], 2)
            self.assertEqual(render_page.call_args.kwargs["last_page"], 2)
            body = response.get_json()
            self.assertEqual(body["pdf_page_number"], 2)
            self.assertEqual(body["pdf_page_count"], 2)
            saved = json.loads(
                (Path(upload_root) / "energy" / "PAGE-TWO" / "recognition.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["pdf_page_number"], 2)
            self.assertEqual(saved["pdf_page_count"], 2)

    def test_ai_recognize_uses_shared_prepared_page_outputs(self):
        from floorplan_page_pipeline import PreparedFloorplanPage

        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                prepared_upload = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "SHARED-PAGE",
                        "raster_file": (self.make_pdf_bytes(["ONE", "TWO"]), "floors.pdf"),
                    },
                    content_type="multipart/form-data",
                ).get_json()
                roi = {
                    "enabled": True,
                    "bbox_px": [0, 0, 2, 2],
                    "confidence": 0.9,
                    "reason": "test",
                }
                prepared_page = PreparedFloorplanPage(
                    page_number=2,
                    page_count=2,
                    render_bgr=np.full((2, 2, 3), 255, dtype=np.uint8),
                    cleaned_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
                    model_view_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                    model_input_512=np.zeros((1, 3, 512, 512), dtype=np.float32),
                    model_input_metadata={
                        "original_size": [2, 2],
                        "resized_size": [512, 512],
                        "padding": [0, 0, 0, 0],
                        "resize_scale": 256.0,
                    },
                    vector_analysis={
                        "status": "completed",
                        "mode": "vector",
                        "timeout_seconds": 15.0,
                        "reason": None,
                    },
                    scale_calibration={"status": "manual_required", "method": "test"},
                    vector_cleanup={
                        "enabled": True,
                        "has_vector_geometry": True,
                        "has_vector_text": False,
                        "ocr": {"status": "not_needed"},
                        "nonstructural_mask": {"enabled": False},
                        "structural_mask": {"enabled": True},
                        "building_roi": roi,
                    },
                    inference_roi=[0, 0, 2, 2],
                    cleanup_mask=np.zeros((2, 2), dtype=np.uint8),
                    structural_support_mask=np.ones((2, 2), dtype=np.uint8) * 255,
                    artifacts={},
                )
                segmenter = MagicMock()
                from floorplan_onnx import FloorplanSegmenterONNX
                real_preprocessor = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
                real_preprocessor.img_size = 512
                segmenter.remove_annotations.side_effect = real_preprocessor.remove_annotations
                segmenter.preprocess_dark_cad.side_effect = real_preprocessor.preprocess_dark_cad
                segmenter.prepare_model_input.side_effect = real_preprocessor.prepare_model_input
                segmenter.predict.return_value = {
                    "mask": np.zeros((2, 2), dtype=np.uint8),
                    "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
                    "stats": {},
                    "geometry": {"walls": [], "windows": [], "doors": []},
                    "room_topology": {
                        "status": "no_closed_rooms", "room_count": 0, "rooms": [],
                        "total_area_px2": 0.0, "total_area_m2": None,
                        "scale_m_per_px": None, "load_geometry_ready": False,
                    },
                    "image_size": [2, 2],
                }
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
                    patch.object(
                        self.server,
                        "prepare_pdf_page",
                        return_value=prepared_page,
                        create=True,
                    ),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "SHARED-PAGE",
                            "pdf_upload_token": prepared_upload["upload_token"],
                            "pdf_page_number": "2",
                        },
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            body = response.get_json()
            self.assertEqual(body["vector_cleanup"]["building_roi"], roi)
            self.assertEqual(body["scale_calibration"]["method"], "test")
            saved_model_input = cv2.imread(
                str(Path(upload_root) / "energy" / "SHARED-PAGE" / "pdf_model_input.png")
            )
            self.assertIsNotNone(saved_model_input)
            self.assertEqual(int(saved_model_input.max()), 0)

    def test_ai_recognize_atomically_writes_artifacts_under_unicode_upload_root(self):
        from floorplan_page_pipeline import PreparedFloorplanPage

        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory) / "网站产物"
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
            try:
                prepared_upload = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "UNICODE-WRITES",
                        "raster_file": (
                            self.make_pdf_bytes(["ONE"]),
                            "floor.pdf",
                        ),
                    },
                    content_type="multipart/form-data",
                ).get_json()
                prepared_page = PreparedFloorplanPage(
                    page_number=1,
                    page_count=1,
                    render_bgr=np.full((2, 2, 3), 240, dtype=np.uint8),
                    cleaned_bgr=np.full((2, 2, 3), 180, dtype=np.uint8),
                    model_view_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                    model_input_512=np.zeros(
                        (1, 3, 512, 512),
                        dtype=np.float32,
                    ),
                    model_input_metadata={
                        "original_size": [2, 2],
                        "resized_size": [512, 512],
                        "padding": [0, 0, 0, 0],
                        "resize_scale": 256.0,
                    },
                    vector_analysis={
                        "status": "completed",
                        "mode": "vector",
                        "timeout_seconds": 15.0,
                        "reason": None,
                    },
                    scale_calibration={
                        "status": "manual_required",
                        "method": "test",
                    },
                    vector_cleanup={
                        "enabled": True,
                        "has_vector_geometry": True,
                        "has_vector_text": False,
                        "structural_mask": {"enabled": True},
                        "building_roi": {
                            "enabled": False,
                            "bbox_px": None,
                        },
                    },
                    inference_roi=None,
                    cleanup_mask=np.array([[255, 0], [0, 0]], dtype=np.uint8),
                    structural_support_mask=np.full(
                        (2, 2),
                        255,
                        dtype=np.uint8,
                    ),
                    artifacts={},
                )
                segmenter = MagicMock()
                segmenter.predict.return_value = {
                    "mask": np.array([[0, 1], [2, 3]], dtype=np.uint8),
                    "raw_model_mask": np.ones((2, 2), dtype=np.uint8),
                    "overlay": np.full((2, 2, 3), 120, dtype=np.uint8),
                    "stats": {},
                    "geometry": {"walls": [], "windows": [], "doors": []},
                    "room_topology": {
                        "status": "no_closed_rooms",
                        "room_count": 0,
                        "rooms": [],
                        "total_area_px2": 0.0,
                        "total_area_m2": None,
                        "load_geometry_ready": False,
                    },
                    "image_size": [2, 2],
                }
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(
                        self.server,
                        "_floorplan_segmenter",
                        segmenter,
                        create=True,
                    ),
                    patch.object(
                        self.server,
                        "prepare_pdf_page",
                        return_value=prepared_page,
                        create=True,
                    ),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "UNICODE-WRITES",
                            "pdf_upload_token": prepared_upload["upload_token"],
                            "pdf_page_number": "1",
                        },
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            target = upload_root / "energy" / "UNICODE-WRITES"
            artifact_names = (
                "building_plan_ai.png",
                "pdf_nonstructural_mask.png",
                "pdf_structural_mask.png",
                "pdf_model_input.png",
                "ai_overlay.jpg",
                "ai_raw_model_mask.png",
                "ai_mask.png",
            )
            for artifact_name in artifact_names:
                with self.subTest(artifact=artifact_name):
                    artifact = target / artifact_name
                    self.assertTrue(artifact.is_file())
                    decoded = cv2.imdecode(
                        np.frombuffer(artifact.read_bytes(), dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED,
                    )
                    self.assertIsNotNone(decoded)
            self.assertTrue(
                (target / "ai_overlay.jpg").read_bytes().startswith(b"\xff\xd8\xff")
            )
            self.assertEqual(
                list(target.glob("*.tmp*")),
                [],
            )

    def test_ai_recognize_rejects_invalid_prepared_pdf_reference_or_page(self):
        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                prepared = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "VALID-REPORT",
                        "raster_file": (self.make_pdf_bytes(["ONE", "TWO"]), "floors.pdf"),
                    },
                    content_type="multipart/form-data",
                ).get_json()
                cases = [
                    ("VALID-REPORT", prepared["upload_token"], "0", "page"),
                    ("VALID-REPORT", prepared["upload_token"], "3", "page"),
                    ("VALID-REPORT", prepared["upload_token"] + "tampered", "1", "token"),
                    ("OTHER-REPORT", prepared["upload_token"], "1", "report"),
                ]
                with patch.object(self.server, "HAS_FLOORPLAN_AI", True):
                    for report_number, token, page_number, expected_error in cases:
                        with self.subTest(report_number=report_number, page_number=page_number):
                            response = self.client.post(
                                "/energy/ai_recognize",
                                data={
                                    "report_number": report_number,
                                    "pdf_upload_token": token,
                                    "pdf_page_number": page_number,
                                },
                            )
                            self.assertEqual(response.status_code, 400)
                            self.assertIn(expected_error, response.get_json()["error"].lower())
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

    def test_vector_pdf_route_applies_cleanup_roi_and_conservative_topology(self):
        from PIL import Image
        from floorplan_page_pipeline import VectorAnalysisOutcome

        with tempfile.TemporaryDirectory() as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                prepared = self.client.post(
                    "/energy/pdf_prepare",
                    data={
                        "report_number": "VECTOR-CLEANUP",
                        "raster_file": (self.make_pdf_bytes(["OUTLINED-TEXT-PAGE"]), "floor.pdf"),
                    },
                    content_type="multipart/form-data",
                ).get_json()
                segmenter = MagicMock()
                from floorplan_onnx import FloorplanSegmenterONNX
                real_preprocessor = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
                real_preprocessor.img_size = 512
                segmenter.remove_annotations.side_effect = real_preprocessor.remove_annotations
                segmenter.preprocess_dark_cad.side_effect = real_preprocessor.preprocess_dark_cad
                segmenter.prepare_model_input.side_effect = real_preprocessor.prepare_model_input
                segmenter.predict.return_value = {
                    "mask": np.zeros((100, 120), dtype=np.uint8),
                    "raw_model_mask": np.ones((100, 120), dtype=np.uint8),
                    "overlay": np.zeros((100, 120, 3), dtype=np.uint8),
                    "stats": {},
                    "geometry": {"walls": [], "windows": [], "doors": []},
                    "room_topology": {
                        "status": "no_closed_rooms", "room_count": 0, "rooms": [],
                        "total_area_px2": 0.0, "total_area_m2": None,
                        "scale_m_per_px": None, "load_geometry_ready": False,
                        "max_gap_px": 12, "min_room_area_px": 500.0,
                    },
                    "image_size": [120, 100],
                    "inference_roi": [10, 10, 110, 90],
                    "topology_repair": {
                        "exterior_repair": {"status": "manual_exterior_wall_required"},
                        "internal_fragments": {"ignored": [], "repaired": [], "ambiguous": []},
                        "manual_exterior_wall_required": True,
                        "manual_review_reasons": ["exterior_not_uniquely_repairable"],
                    },
                }
                page_data = {
                    "is_vector_pdf": True,
                    "has_vector_geometry": True,
                    "has_vector_text": False,
                    "page_size_pt": [120.0, 100.0],
                    "render_size_px": [120, 100],
                    "text_spans": [], "segments": [], "styled_edges": [],
                }
                vector_mask = np.zeros((100, 120), dtype=np.uint8)
                vector_mask[40:45, 30:90] = 255
                structural_mask = np.zeros((100, 120), dtype=np.uint8)
                structural_mask[10:90, 10:110] = 255
                cleanup_evidence = {
                    "enabled": True, "black_edge_count": 50,
                    "gray_edge_count": 20, "masked_pixels": 300,
                }
                roi = {
                    "enabled": True, "bbox_px": [10, 10, 110, 90],
                    "confidence": 0.9, "reason": "dense_black_vector_cluster",
                }
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(self.server, "HAS_VECTOR_PDF_SCALE", True),
                    patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
                    patch(
                        "floorplan_page_pipeline._run_vector_analysis",
                        return_value=VectorAnalysisOutcome(
                            page_data,
                            {
                                "status": "completed",
                                "mode": "vector",
                                "timeout_seconds": 15.0,
                                "reason": None,
                            },
                        ),
                    ) as run_analysis,
                    patch("floorplan_page_pipeline.build_nonstructural_vector_mask", return_value=(vector_mask, cleanup_evidence)),
                    patch("floorplan_page_pipeline.build_structural_vector_mask", return_value=(structural_mask, {"enabled": True, "dark_edge_count": 50, "masked_pixels": 8000})),
                    patch("floorplan_page_pipeline.detect_building_roi", return_value=roi),
                    patch("floorplan_page_pipeline.convert_from_path", return_value=[Image.new("RGB", (120, 100), "white")]) as render_page,
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "VECTOR-CLEANUP",
                            "pdf_upload_token": prepared["upload_token"],
                            "pdf_page_number": "1",
                        },
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertEqual(run_analysis.call_args.kwargs["dpi"], 100)
            self.assertEqual(run_analysis.call_args.kwargs["timeout_seconds"], 30.0)
            self.assertEqual(
                [call.kwargs["dpi"] for call in render_page.call_args_list],
                [100],
            )
            kwargs = segmenter.predict.call_args.kwargs
            self.assertEqual(kwargs["inference_roi"], [10, 10, 110, 90])
            self.assertTrue(kwargs["preserve_full_context"])
            self.assertEqual(kwargs["topology_max_gap_px"], 12)
            self.assertEqual(kwargs["topology_min_room_area_px"], 500.0)
            repair_context = kwargs["topology_repair_context"]
            self.assertEqual(repair_context["building_roi"], [10, 10, 110, 90])
            self.assertTrue(np.array_equal(repair_context["structural_support_mask"], structural_mask))
            body = response.get_json()
            self.assertTrue(body["vector_cleanup"]["nonstructural_mask"]["enabled"])
            self.assertEqual(body["vector_cleanup"]["building_roi"]["bbox_px"], [10, 10, 110, 90])
            report_dir = Path(upload_root) / "energy" / "VECTOR-CLEANUP"
            self.assertTrue((report_dir / "pdf_nonstructural_mask.png").exists())
            self.assertTrue((report_dir / "pdf_model_input.png").exists())
            self.assertTrue((report_dir / "pdf_building_roi.json").exists())
            self.assertTrue((report_dir / "ai_raw_model_mask.png").exists())
            self.assertEqual(body["topology_repair"]["exterior_repair"]["status"], "manual_exterior_wall_required")

    def test_vector_pdf_fusion_returns_complete_exterior_review_summary(self):
        fusion = {
            "status": "evaluable",
            "summary": {},
            "reason_codes": [],
            "exterior_summary": {
                "footprint_status": "closed",
                "unresolved_gap_count": 1,
            },
            "exterior_topology": {
                "area_px2": 5000.0,
                "perimeter_px": 300.0,
                "bridges": [
                    {"bridge_id": "repair-1", "bridge_type": "small_gap_repair"},
                    {"bridge_id": "opening-bridge-1", "bridge_type": "opening_bridge"},
                ],
                "unresolved": [{"gap_id": "gap-1"}],
            },
            "opening_candidates": {
                "accepted_openings": [
                    {"opening_id": "door-1", "kind": "door", "width_px": 30.0},
                    {"opening_id": "window-1", "kind": "window", "width_px": 20.0},
                ],
            },
        }
        with (
            patch.object(self.server, "HAS_VECTOR_PDF_FUSION", True),
            patch.object(self.server, "_vector_pdf_fusion_config", object()),
            patch.object(
                self.server,
                "_validated_prepared_pdf",
                return_value=(Path("prepared.pdf"), 2, 3),
            ),
            patch.object(self.server, "analyze_vector_pdf_page", return_value=fusion),
            patch.object(self.server, "_sha256_file", return_value="a" * 64),
            patch.object(self.server.cv2, "imread", return_value=None),
        ):
            response = self.client.post(
                "/energy/vector_pdf_fusion",
                data={
                    "report_number": "EXT-REVIEW",
                    "pdf_upload_token": "token",
                    "pdf_page_number": "2",
                    "recognition_mode": "full_page",
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        summary = response.get_json()["exterior_summary"]
        self.assertEqual(summary["area_px2"], 5000.0)
        self.assertEqual(summary["perimeter_px"], 300.0)
        self.assertEqual(summary["door_count"], 1)
        self.assertEqual(summary["door_total_width_px"], 30.0)
        self.assertEqual(summary["window_count"], 1)
        self.assertEqual(summary["window_total_width_px"], 20.0)
        self.assertEqual(summary["small_repair_count"], 1)
        self.assertEqual(summary["unresolved_gap_count"], 1)

    def post_and_capture_params(self, payload):
        result = {
            "success": True,
            "summary": {"total_energy_kwh": 0, "eui": 0, "rating": "A", "rating_label": "test"},
        }
        recognition = {
            "model": {"version": self.server.FLOORPLAN_MODEL_VERSION},
            "preprocessing": {"requested": "auto", "use_preprocessing": True},
            "image_size": [2, 2],
            "geometry": {"walls": [], "windows": [], "doors": []},
        }
        segmenter = MagicMock()
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = None

        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(self.server, "HAS_ENERGY_CALC", True),
            patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
            patch.object(self.server, "_load_recognition_payload", return_value=recognition),
            patch.object(self.server.energy_calc, "calculate_energy", return_value=result) as calculate,
            patch.object(self.server, "get_db_connection", return_value=connection),
        ):
            response = self.client.post("/energy/ai_simulate", json={"report_number": "TDD", **payload})

        params = calculate.call_args.args[0] if calculate.called else None
        return response, params

    def post_exterior_energy(self, recognition, payload=None):
        result = {
            "success": True,
            "summary": {
                "total_energy_kwh": 0,
                "eui": 0,
                "rating": "A",
                "rating_label": "test",
            },
        }
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = None
        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(self.server, "HAS_ENERGY_CALC", True),
            patch.object(
                self.server, "_exterior_report_lock",
                return_value=nullcontext(), create=True,
            ),
            patch.object(
                self.server, "_require_current_exterior_generation",
                return_value={
                    "generation": "test-generation",
                    "status": "ready",
                    "topology_sha256": "a" * 64,
                },
                create=True,
            ),
            patch.object(self.server, "_load_recognition_payload", return_value=recognition),
            patch.object(self.server.energy_calc, "calculate_energy", return_value=result) as calculate,
            patch.object(self.server, "get_db_connection", return_value=connection),
        ):
            response = self.client.post(
                "/energy/ai_simulate",
                json={"report_number": "EXT-ENERGY", **(payload or {})},
            )
        return response, calculate

    @staticmethod
    def exterior_recognition(*, confirmed=True, room_topology=None, perimeter_m=40.0):
        return {
            "model": {"version": "exterior-test"},
            "preprocessing": {"requested": "vector_pdf_fusion", "use_preprocessing": False},
            "image_size": [100, 100],
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": room_topology or {
                "status": "exterior_only",
                "room_count": 0,
                "rooms": [],
                "total_area_px2": 0.0,
                "load_geometry_ready": False,
            },
            "exterior_topology": {
                "area_m2": 100.0,
                "perimeter_m": perimeter_m,
                "confirmed": confirmed,
                "load_geometry_ready": confirmed,
            },
            "openings": [
                {"opening_id": "door-1", "kind": "door", "width_m": 2.0},
                {"opening_id": "window-1", "kind": "window", "width_m": 8.0},
            ],
        }

    def post_persisted_exterior_energy(self, recognition, marker):
        result = {
            "success": True,
            "summary": {"total_energy_kwh": 0, "eui": 0, "rating": "A", "rating_label": "test"},
        }
        temporary = tempfile.TemporaryDirectory()
        upload_root = Path(temporary.name)
        report_dir = upload_root / "energy" / "EXT-ENERGY"
        report_dir.mkdir(parents=True)
        (report_dir / "recognition.json").write_text(
            json.dumps(recognition), encoding="utf-8",
        )
        if marker is not None:
            (report_dir / "exterior_generation.json").write_text(
                json.dumps(marker), encoding="utf-8",
            )
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = None
        previous_upload = self.server.app.config["UPLOAD_FOLDER"]
        self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
        try:
            with (
                patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                patch.object(self.server, "HAS_ENERGY_CALC", True),
                patch.object(self.server, "HAS_DESIGN_LOAD_CALC", False),
                patch.object(self.server.energy_calc, "calculate_energy", return_value=result) as calculate,
                patch.object(self.server, "get_db_connection", return_value=connection),
            ):
                response = self.client.post(
                    "/energy/ai_simulate",
                    json={"report_number": "EXT-ENERGY", "height": 3.0, "floors": 2},
                )
        finally:
            self.server.app.config["UPLOAD_FOLDER"] = previous_upload
            temporary.cleanup()
        return response, calculate

    @staticmethod
    def bind_exterior_generation(recognition, *, generation="generation-a", topology_sha256="a" * 64):
        recognition = json.loads(json.dumps(recognition))
        recognition["schema_version"] = 1
        recognition["report_number"] = "EXT-ENERGY"
        recognition["exterior_generation"] = {
            "generation": generation,
            "topology_sha256": topology_sha256,
        }
        return recognition

    @staticmethod
    def exterior_marker(*, status="ready", generation="generation-a", topology_sha256="a" * 64):
        return {
            "format": "pdf-exterior-generation/1",
            "report_number": "EXT-ENERGY",
            "generation": generation,
            "status": status,
            "topology_sha256": topology_sha256 if status == "ready" else None,
        }

    def test_energy_route_rejects_confirmed_exterior_when_generation_is_not_current(self):
        recognition = self.bind_exterior_generation(self.exterior_recognition())
        cases = (
            ("missing marker", None),
            ("running generation", self.exterior_marker(status="running", generation="generation-b")),
            ("failed generation", self.exterior_marker(status="failed", generation="generation-b")),
            ("generation mismatch", self.exterior_marker(generation="generation-b")),
            ("hash mismatch", self.exterior_marker(topology_sha256="b" * 64)),
        )
        for label, marker in cases:
            with self.subTest(label=label):
                response, calculate = self.post_persisted_exterior_energy(recognition, marker)
                self.assertEqual(response.status_code, 409, response.get_json())
                self.assertIn("generation", response.get_json()["error"].lower())
                calculate.assert_not_called()

    def test_stale_exterior_uses_existing_ready_room_fallback(self):
        room_topology = {
            "status": "closed",
            "room_count": 1,
            "rooms": [{"id": "room-1", "area_px2": 400.0}],
            "total_area_px2": 400.0,
            "scale_m_per_px": 0.5,
            "load_geometry_ready": True,
        }
        recognition = self.bind_exterior_generation(
            self.exterior_recognition(room_topology=room_topology),
        )
        recognition["scale_calibration"] = {
            "status": "confirmed",
            "method": "legacy-room-test",
            "scale_m_per_px": 0.5,
        }
        response, calculate = self.post_persisted_exterior_energy(
            recognition,
            self.exterior_marker(status="failed", generation="generation-b"),
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["floor_area_source"], "room_polygons")
        self.assertEqual(calculate.call_args.args[0]["geometry"]["floor_area_m2"], 100.0)

    def test_energy_route_uses_confirmed_exterior_and_separate_opening_heights(self):
        response, calculate = self.post_exterior_energy(
            self.exterior_recognition(),
            {
                "height": 3.0,
                "floors": 2,
                "door_height_m": 2.1,
                "window_height_m": 1.5,
                "door_repeat_count": 1,
                "window_repeat_count": 2,
            },
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        geometry = calculate.call_args.args[0]["geometry"]
        self.assertEqual(geometry["floor_area_m2"], 100.0)
        self.assertEqual(geometry["roof_area_m2"], 100.0)
        self.assertEqual(geometry["door_area_m2"], 4.2)
        self.assertEqual(geometry["window_area_m2"], 24.0)
        self.assertEqual(geometry["wall_area_m2"], 211.8)
        self.assertEqual(response.get_json()["floor_area_source"], "confirmed_exterior_footprint")

    def test_energy_route_defaults_exterior_repeats_to_door_once_and_windows_all_floors(self):
        response, calculate = self.post_exterior_energy(
            self.exterior_recognition(),
            {
                "height": 3.0,
                "floors": 2,
                "door_height_m": 2.1,
                "window_height_m": 1.5,
            },
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        geometry = calculate.call_args.args[0]["geometry"]
        self.assertEqual(geometry["door_area_m2"], 4.2)
        self.assertEqual(geometry["window_area_m2"], 24.0)

    def test_unconfirmed_exterior_cannot_calculate_without_an_existing_fallback(self):
        response, calculate = self.post_exterior_energy(
            self.exterior_recognition(confirmed=False),
            {"height": 3.0, "floors": 2},
        )

        self.assertEqual(response.status_code, 400, response.get_json())
        calculate.assert_not_called()

    def test_unconfirmed_exterior_does_not_preempt_old_confirmed_room_topology(self):
        room_topology = {
            "status": "closed",
            "room_count": 1,
            "rooms": [{"id": "room-1", "area_px2": 400.0}],
            "total_area_px2": 400.0,
            "scale_m_per_px": 0.5,
            "load_geometry_ready": True,
        }
        recognition = self.exterior_recognition(
            confirmed=False,
            room_topology=room_topology,
        )
        recognition["scale_calibration"] = {
            "status": "confirmed",
            "method": "legacy-room-test",
            "scale_m_per_px": 0.5,
        }

        response, calculate = self.post_exterior_energy(
            recognition,
            {"height": 3.0, "floors": 2},
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["floor_area_source"], "room_polygons")
        self.assertEqual(calculate.call_args.args[0]["geometry"]["floor_area_m2"], 100.0)

    def test_energy_route_rejects_exterior_openings_larger_than_gross_wall(self):
        response, calculate = self.post_exterior_energy(
            self.exterior_recognition(perimeter_m=4.0),
            {
                "height": 3.0,
                "floors": 1,
                "door_height_m": 3.0,
                "window_height_m": 1.0,
            },
        )

        self.assertEqual(response.status_code, 400, response.get_json())
        self.assertIn("gross exterior wall area", response.get_json()["error"])
        calculate.assert_not_called()

    def test_energy_route_strictly_validates_exterior_repeat_counts(self):
        for field, value in (
            ("door_repeat_count", 1.5),
            ("door_repeat_count", 3),
            ("window_repeat_count", 1.5),
            ("window_repeat_count", 3),
        ):
            with self.subTest(field=field, value=value):
                response, calculate = self.post_exterior_energy(
                    self.exterior_recognition(),
                    {"height": 3.0, "floors": 2, field: value},
                )
                self.assertEqual(response.status_code, 400, response.get_json())
                self.assertIn(field, response.get_json()["error"])
                calculate.assert_not_called()

    def post_with_actual_calculator(self, payload):
        recognition = {
            "model": {"version": self.server.FLOORPLAN_MODEL_VERSION},
            "preprocessing": {"requested": "auto", "use_preprocessing": True},
            "image_size": [2, 2],
            "geometry": {"walls": [], "windows": [], "doors": []},
        }
        segmenter = MagicMock()
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = None
        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(self.server, "HAS_ENERGY_CALC", True),
            patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
            patch.object(self.server, "_load_recognition_payload", return_value=recognition),
            patch.object(self.server, "get_db_connection", return_value=connection),
        ):
            return self.client.post("/energy/ai_simulate", json={"report_number": "V2", **payload})

    def test_ai_recognize_persists_recognition_json(self):
        with tempfile.TemporaryDirectory(dir=r"D:\Projects") as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            segmenter = MagicMock()
            segmenter.predict.return_value = {
                "mask": np.array([[0, 1], [2, 3]], dtype=np.uint8),
                "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
                "stats": {
                    "background": {"pixels": 1, "percentage": 25.0, "color": "#282828"},
                    "wall": {"pixels": 1, "percentage": 25.0, "color": "#e74c3c"},
                    "window": {"pixels": 1, "percentage": 25.0, "color": "#3498db"},
                    "door": {"pixels": 1, "percentage": 25.0, "color": "#2ecc71"},
                },
                "geometry": {
                    "walls": [{"pts": [[0.0, 0.0], [2.0, 0.0]], "area": 2.0, "bbox": [0, 0, 2, 1]}],
                    "windows": [],
                    "doors": [],
                },
                "room_topology": {
                    "status": "closed",
                    "room_count": 1,
                    "closure_applied": True,
                    "rooms": [{"id": "room-1", "polygon_px": [[0, 0], [1, 0], [1, 1]], "area_px2": 1.0, "area_m2": None}],
                    "total_area_px2": 1.0,
                    "total_area_m2": None,
                    "scale_m_per_px": None,
                    "load_geometry_ready": False,
                },
                "image_size": [2, 2],
            }
            try:
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
                    patch.object(self.server.cv2, "imread", return_value=np.zeros((2, 2, 3), dtype=np.uint8)),
                    patch.object(self.server.cv2, "imwrite", return_value=True),
                    patch.object(
                        self.server.cv2,
                        "imencode",
                        return_value=(True, np.array([1, 2, 3], dtype=np.uint8)),
                    ),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "JSON-1",
                            "preprocessing": "auto",
                            "raster_file": (io.BytesIO(b"fake image"), "plan.png"),
                        },
                        content_type="multipart/form-data",
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            recognition_path = Path(upload_root) / "energy" / "JSON-1" / "recognition.json"
            self.assertTrue(recognition_path.exists())
            recognition = json.loads(recognition_path.read_text(encoding="utf-8"))
            self.assertEqual(recognition["schema_version"], 1)
            self.assertEqual(recognition["model"]["version"], self.server.FLOORPLAN_MODEL_VERSION)
            self.assertEqual(recognition["preprocessing"]["requested"], "auto")
            self.assertEqual(recognition["image_size"], [2, 2])
            self.assertEqual(recognition["geometry_summary"], {"walls": 1, "windows": 0, "doors": 0})
            self.assertEqual(recognition["room_topology"]["room_count"], 1)
            self.assertEqual(recognition["mask"]["path"], "ai_mask.png")
            self.assertEqual(response.get_json()["room_topology"]["room_count"], 1)
            self.assertIn("recognition", response.get_json())

    def test_vector_pdf_recognition_persists_confirmed_scale_and_masks_dimensions(self):
        from reportlab.pdfgen import canvas
        from floorplan_onnx import FloorplanSegmenterONNX

        pdf_bytes = io.BytesIO()
        pdf = canvas.Canvas(pdf_bytes, pagesize=(400, 300))
        pdf.setLineWidth(0.5)
        pdf.line(40, 260, 360, 260)
        pdf.line(40, 250, 40, 280)
        pdf.line(360, 250, 360, 280)
        pdf.drawString(178, 266, "40600")
        vertical_span = 320 * 18.6 / 40.6
        lower = 150 - vertical_span / 2
        upper = 150 + vertical_span / 2
        pdf.line(25, lower, 25, upper)
        pdf.line(15, lower, 35, lower)
        pdf.line(15, upper, 35, upper)
        pdf.saveState()
        pdf.translate(12, 124)
        pdf.rotate(90)
        pdf.drawString(0, 0, "18600")
        pdf.restoreState()
        pdf.rect(80, 80, 240, 140)
        pdf.save()
        pdf_bytes.seek(0)

        segmenter = MagicMock()
        real_preprocessor = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        real_preprocessor.img_size = 512
        segmenter.remove_annotations.side_effect = real_preprocessor.remove_annotations
        segmenter.preprocess_dark_cad.side_effect = real_preprocessor.preprocess_dark_cad
        segmenter.prepare_model_input.side_effect = real_preprocessor.prepare_model_input
        segmenter.predict.return_value = {
            "mask": np.zeros((2, 2), dtype=np.uint8),
            "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
            "stats": {},
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {
                "status": "closed",
                "room_count": 1,
                "rooms": [{"id": "room-1", "area_px2": 100.0, "area_m2": None}],
                "total_area_px2": 100.0,
                "total_area_m2": None,
                "scale_m_per_px": None,
                "load_geometry_ready": False,
            },
            "image_size": [2, 2],
        }

        with tempfile.TemporaryDirectory(dir=r"D:\Projects") as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
                    patch.dict(os.environ, {
                        "POPPLER_PATH": r"C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
                    }),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "VECTOR-PDF",
                            "preprocessing": "auto",
                            "raster_file": (pdf_bytes, "plan.pdf"),
                        },
                        content_type="multipart/form-data",
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            body = response.get_json()
            self.assertEqual(body["scale_calibration"]["status"], "confirmed")
            self.assertEqual(body["scale_calibration"]["horizontal"]["text"], "40600")
            self.assertEqual(body["scale_calibration"]["vertical"]["text"], "18600")
            self.assertTrue(body["room_topology"]["load_geometry_ready"])
            expected_area = 100.0 * body["scale_calibration"]["scale_m_per_px"] ** 2
            self.assertAlmostEqual(body["room_topology"]["total_area_m2"], expected_area)
            recognition_path = Path(upload_root) / "energy" / "VECTOR-PDF" / "recognition.json"
            saved = json.loads(recognition_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["scale_calibration"]["status"], "confirmed")
            self.assertAlmostEqual(saved["room_topology"]["total_area_m2"], expected_area)

            cleaned = segmenter.predict.call_args.args[0]
            self.assertGreater(cleaned[56, 250].min(), 240)
            self.assertLess(cleaned[111, 111].min(), 100)

    def test_vector_pdf_without_text_uses_ocr_scale_fallback(self):
        from reportlab.pdfgen import canvas
        from floorplan_onnx import FloorplanSegmenterONNX

        pdf_bytes = io.BytesIO()
        pdf = canvas.Canvas(pdf_bytes, pagesize=(400, 300))
        pdf.setLineWidth(0.5)
        pdf.line(40, 260, 360, 260)
        pdf.line(40, 250, 40, 280)
        pdf.line(360, 250, 360, 280)
        vertical_span = 320 * 18.6 / 40.6
        lower = 150 - vertical_span / 2
        upper = 150 + vertical_span / 2
        pdf.line(25, lower, 25, upper)
        pdf.line(15, lower, 35, lower)
        pdf.line(15, upper, 35, upper)
        pdf.rect(80, 80, 240, 140)
        pdf.save()
        pdf_bytes.seek(0)

        ocr_spans = [
            {
                "text": "40600",
                "bbox_pt": [178, 22, 222, 32],
                "center_pt": [200, 27],
                "direction": "horizontal",
                "font_size": 10,
                "source": "rapidocr",
                "confidence": 0.98,
            },
            {
                "text": "18600",
                "bbox_pt": [12, 125, 22, 175],
                "center_pt": [17, 150],
                "direction": "vertical",
                "font_size": 10,
                "source": "rapidocr",
                "confidence": 0.96,
            },
        ]
        segmenter = MagicMock()
        real_preprocessor = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        real_preprocessor.img_size = 512
        segmenter.remove_annotations.side_effect = real_preprocessor.remove_annotations
        segmenter.preprocess_dark_cad.side_effect = real_preprocessor.preprocess_dark_cad
        segmenter.prepare_model_input.side_effect = real_preprocessor.prepare_model_input
        segmenter.predict.return_value = {
            "mask": np.zeros((2, 2), dtype=np.uint8),
            "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
            "stats": {},
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {
                "status": "closed",
                "room_count": 1,
                "rooms": [{"id": "room-1", "area_px2": 100.0, "area_m2": None}],
                "total_area_px2": 100.0,
                "total_area_m2": None,
                "scale_m_per_px": None,
                "load_geometry_ready": False,
            },
            "image_size": [2, 2],
        }

        with tempfile.TemporaryDirectory(dir=r"D:\Projects") as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
                    patch(
                        "floorplan_page_pipeline.extract_numeric_text_spans",
                        return_value=(ocr_spans, {
                            "status": "completed",
                            "candidate_count": 2,
                            "accepted_count": 2,
                            "min_confidence": 0.60,
                        }),
                        create=True,
                    ) as extract_ocr,
                    patch(
                        "floorplan_page_pipeline.build_dimension_annotation_mask",
                        return_value=np.zeros((417, 556), dtype=np.uint8),
                    ) as dimension_mask,
                    patch.dict(os.environ, {
                        "POPPLER_PATH": r"C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
                    }),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "VECTOR-OCR-PDF",
                            "preprocessing": "auto",
                            "raster_file": (pdf_bytes, "outline-text-plan.pdf"),
                        },
                        content_type="multipart/form-data",
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        extract_ocr.assert_called_once()
        dimension_mask.assert_not_called()
        self.assertEqual(body["scale_calibration"]["status"], "confirmed")
        self.assertEqual(body["scale_calibration"]["text_source"], "rapidocr")
        self.assertEqual(body["scale_calibration"]["horizontal"]["text"], "40600")
        self.assertEqual(body["scale_calibration"]["vertical"]["text"], "18600")
        self.assertEqual(body["vector_cleanup"]["ocr"]["accepted_count"], 2)

    def test_ai_simulate_reuses_saved_recognition_without_second_prediction(self):
        with tempfile.TemporaryDirectory(dir=r"D:\Projects") as upload_root:
            report_dir = Path(upload_root) / "energy" / "REUSE"
            report_dir.mkdir(parents=True)
            recognition = {
                "schema_version": 1,
                "model": {"name": self.server.FLOORPLAN_MODEL_NAME, "version": self.server.FLOORPLAN_MODEL_VERSION},
                "preprocessing": {"requested": "auto", "use_preprocessing": True},
                "image_size": [10, 10],
                "stats": {},
                "geometry": {
                    "walls": [{"pts": [[0.0, 0.0], [10.0, 0.0]], "area": 20.0, "bbox": [0, 0, 10, 2]}],
                    "windows": [{"pts": [[0.0, 0.0], [5.0, 0.0]], "area": 5.0, "bbox": [0, 0, 5, 1]}],
                    "doors": [{"pts": [[0.0, 0.0], [1.0, 0.0]], "area": 4.0, "bbox": [0, 0, 1, 1]}],
                },
                "geometry_summary": {"walls": 1, "windows": 1, "doors": 1},
                "room_topology": {
                    "status": "closed",
                    "room_count": 1,
                    "rooms": [{"id": "room-1", "area_px2": 200.0, "area_m2": None}],
                    "total_area_px2": 200.0,
                    "total_area_m2": None,
                    "scale_m_per_px": None,
                    "load_geometry_ready": False,
                },
                "pixel_lengths": {"wall_px": 10.0, "window_px": 5.0},
                "mask": {"path": "ai_mask.png", "shape": [10, 10]},
                "artifacts": {"original": "building_plan_ai.png", "overlay": "ai_overlay.jpg", "mask": "ai_mask.png"},
            }
            (report_dir / "recognition.json").write_text(json.dumps(recognition), encoding="utf-8")
            result = {
                "success": True,
                "summary": {"total_energy_kwh": 0, "eui": 0, "rating": "A", "rating_label": "test"},
            }
            segmenter = MagicMock()
            connection = MagicMock()
            connection.execute.return_value.fetchone.return_value = None
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(self.server, "HAS_ENERGY_CALC", True),
                    patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
                    patch.object(self.server.energy_calc, "calculate_energy", return_value=result) as calculate,
                    patch.object(self.server, "get_db_connection", return_value=connection),
                ):
                    response = self.client.post(
                        "/energy/ai_simulate",
                        json={"report_number": "REUSE", "scale": 0.5, "height": 3.0, "floor_area_m2": 100},
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            segmenter.predict.assert_not_called()
            params = calculate.call_args.args[0]
            self.assertEqual(params["geometry"]["floor_area_m2"], 50.0)
            self.assertEqual(response.get_json()["floor_area_source"], "room_polygons")
            self.assertAlmostEqual(params["geometry"]["wall_area_m2"], 15.0)
            self.assertAlmostEqual(params["geometry"]["window_area_m2"], 7.5)
            self.assertAlmostEqual(params["geometry"]["door_area_m2"], 1.0)
            self.assertEqual(response.get_json()["recognition_source"], "recognition_json")

    def test_ai_simulate_uses_manual_area_when_exterior_repair_is_required(self):
        recognition = {
            "schema_version": 1,
            "model": {"version": self.server.FLOORPLAN_MODEL_VERSION},
            "image_size": [100, 100],
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {
                "status": "closed", "room_count": 1,
                "rooms": [{"id": "room-1", "area_px2": 200.0, "area_m2": None}],
                "total_area_px2": 200.0, "load_geometry_ready": False,
            },
            "topology_repair": {"manual_exterior_wall_required": True},
            "pixel_lengths": {"wall_px": 0, "window_px": 0},
        }
        calculation = {
            "success": True,
            "summary": {"total_energy_kwh": 0, "eui": 0, "rating": "A", "rating_label": "test"},
        }
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = None
        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(self.server, "HAS_ENERGY_CALC", True),
            patch.object(self.server, "_load_recognition_payload", return_value=recognition),
            patch.object(self.server.energy_calc, "calculate_energy", return_value=calculation) as calculate,
            patch.object(self.server, "get_db_connection", return_value=connection),
        ):
            response = self.client.post(
                "/energy/ai_simulate",
                json={"report_number": "MANUAL-AREA", "scale": 0.1, "floor_area_m2": 100},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["floor_area_source"], "manual")
        self.assertEqual(calculate.call_args.args[0]["geometry"]["floor_area_m2"], 100.0)

    def test_ai_simulate_uses_confirmed_persisted_scale_instead_of_frontend_default(self):
        recognition = {
            "schema_version": 1,
            "model": {"name": self.server.FLOORPLAN_MODEL_NAME, "version": self.server.FLOORPLAN_MODEL_VERSION},
            "preprocessing": {"requested": "auto", "use_preprocessing": True},
            "image_size": [100, 100],
            "stats": {},
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {
                "status": "closed",
                "room_count": 1,
                "rooms": [{"id": "room-1", "area_px2": 200.0, "area_m2": 2.0}],
                "total_area_px2": 200.0,
                "total_area_m2": 2.0,
                "scale_m_per_px": 0.1,
                "load_geometry_ready": True,
            },
            "scale_calibration": {
                "status": "confirmed",
                "method": "vector_pdf_overall_dimensions",
                "scale_m_per_px": 0.1,
                "confidence": 0.95,
            },
        }
        result = {
            "success": True,
            "summary": {"total_energy_kwh": 0, "eui": 0, "rating": "A", "rating_label": "test"},
        }
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = None
        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(self.server, "HAS_ENERGY_CALC", True),
            patch.object(self.server, "_load_recognition_payload", return_value=recognition),
            patch.object(self.server.energy_calc, "calculate_energy", return_value=result) as calculate,
            patch.object(self.server, "get_db_connection", return_value=connection),
        ):
            response = self.client.post(
                "/energy/ai_simulate",
                json={"report_number": "AUTO-SCALE", "scale": 0.5, "floor_area_m2": 100},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertAlmostEqual(calculate.call_args.args[0]["geometry"]["floor_area_m2"], 2.0)
        self.assertEqual(response.get_json()["scale_source"], "vector_pdf_overall_dimensions")

    def test_manual_two_point_scale_calibration_is_saved_and_updates_room_area(self):
        with tempfile.TemporaryDirectory(dir=r"D:\Projects") as upload_root:
            report_dir = Path(upload_root) / "energy" / "MANUAL-SCALE"
            report_dir.mkdir(parents=True)
            recognition = {
                "schema_version": 1,
                "model": {"name": self.server.FLOORPLAN_MODEL_NAME, "version": self.server.FLOORPLAN_MODEL_VERSION},
                "preprocessing": {"requested": "auto", "use_preprocessing": True},
                "image_size": [200, 100],
                "geometry": {"walls": [], "windows": [], "doors": []},
                "room_topology": {
                    "status": "closed",
                    "room_count": 1,
                    "rooms": [{"id": "room-1", "area_px2": 500.0, "area_m2": None}],
                    "total_area_px2": 500.0,
                    "total_area_m2": None,
                    "scale_m_per_px": None,
                    "load_geometry_ready": False,
                },
            }
            (report_dir / "recognition.json").write_text(json.dumps(recognition), encoding="utf-8")
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                response = self.client.post(
                    "/energy/scale_calibration",
                    json={
                        "report_number": "MANUAL-SCALE",
                        "point_a": [10, 10],
                        "point_b": [110, 10],
                        "actual_length": 10000,
                        "unit": "mm",
                    },
                )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            body = response.get_json()
            self.assertEqual(body["scale_calibration"]["method"], "manual_two_point")
            self.assertAlmostEqual(body["scale_calibration"]["scale_m_per_px"], 0.1)
            self.assertAlmostEqual(body["room_topology"]["total_area_m2"], 5.0)
            saved = json.loads((report_dir / "recognition.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(saved["scale_calibration"]["scale_m_per_px"], 0.1)

    def test_manual_scale_does_not_enable_load_when_exterior_needs_wall_repair(self):
        recognition = {
            "schema_version": 1,
            "model": {"version": self.server.FLOORPLAN_MODEL_VERSION},
            "image_size": [200, 100],
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {
                "status": "closed", "room_count": 1,
                "rooms": [{"id": "room-1", "area_px2": 500.0, "area_m2": None}],
                "total_area_px2": 500.0, "total_area_m2": None,
                "scale_m_per_px": None, "load_geometry_ready": False,
            },
            "topology_repair": {
                "manual_exterior_wall_required": True,
                "manual_review_reasons": ["exterior_not_uniquely_repairable"],
            },
        }
        with (
            patch.object(self.server, "_load_recognition_payload", return_value=recognition),
            patch.object(self.server, "_save_recognition_payload"),
        ):
            response = self.client.post(
                "/energy/scale_calibration",
                json={
                    "report_number": "MANUAL-REPAIR",
                    "point_a": [10, 10], "point_b": [110, 10],
                    "actual_length": 10, "unit": "m",
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertFalse(response.get_json()["room_topology"]["load_geometry_ready"])

    def test_manual_scale_calibration_rejects_invalid_points_length_and_unit(self):
        recognition = {
            "schema_version": 1,
            "model": {"version": self.server.FLOORPLAN_MODEL_VERSION},
            "image_size": [200, 100],
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {"room_count": 0, "rooms": [], "total_area_px2": 0},
        }
        with patch.object(self.server, "_load_recognition_payload", return_value=recognition):
            invalid_payloads = [
                {"point_a": [10, 10], "point_b": [10, 10], "actual_length": 10, "unit": "m"},
                {"point_a": [10, 10], "point_b": [110, 10], "actual_length": 0, "unit": "m"},
                {"point_a": [10, 10], "point_b": [110, 10], "actual_length": 10, "unit": "cm"},
                {"point_a": [-1, 10], "point_b": [110, 10], "actual_length": 10, "unit": "m"},
            ]
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    response = self.client.post(
                        "/energy/scale_calibration",
                        json={"report_number": "INVALID", **payload},
                    )
                    self.assertEqual(response.status_code, 400, response.get_json())

    def test_ai_simulate_requires_saved_recognition_json(self):
        with tempfile.TemporaryDirectory(dir=r"D:\Projects") as upload_root:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_root
            segmenter = MagicMock()
            try:
                with (
                    patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
                ):
                    response = self.client.post("/energy/ai_simulate", json={"report_number": "MISSING"})
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 404, response.get_json())
            self.assertIn("recognition.json", response.get_json()["error"])
            segmenter.predict.assert_not_called()

    def test_server_imports_root_energy_calculator(self):
        expected = Path(self.server.__file__).with_name("energy_calc.py").resolve()
        actual = Path(self.server.energy_calc.__file__).resolve()
        self.assertEqual(actual, expected)

    def test_real_route_returns_v2_thermal_and_electricity_blocks(self):
        response = self.post_with_actual_calculator({
            "floor_area_m2": 100,
            "city_id": "xian",
            "summer_outdoor_relative_humidity_percent": 60,
            "summer_indoor_relative_humidity_percent": 50,
        })

        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertIn("annual_thermal_demand_kwh_th", result)
        self.assertIn("annual_electricity_kwh", result)
        self.assertIn("design_loads", result)
        self.assertGreater(result["design_loads"]["heating_design_load"]["total_w"], 0)
        self.assertGreater(result["design_loads"]["cooling_design_load"]["total_w"], 0)
        self.assertEqual(
            result["design_loads"]["derived_inputs"]["summer_outdoor_enthalpy_source"],
            "temperature_relative_humidity",
        )

    def test_route_rejects_non_numeric_non_finite_and_out_of_range_annual_inputs(self):
        invalid_cases = [
            {"area_per_person_m2": "bad"},
            {"area_per_person_m2": "nan"},
            {"fresh_air_m3h_per_person": -1},
            {"daily_operation_hours": 25},
            {"annual_operation_days": 366},
            {"heat_recovery_efficiency": 101},
            {"heat_recovery_efficiency_percent": -1},
            {"infiltration_ach": -0.1},
            {"fan_power_w_per_m3h": "inf"},
            {"heating_seasonal_efficiency": 0},
            {"cooling_seasonal_efficiency": "nan"},
            {"winter_design_temperature_c": "nan"},
            {"summer_design_temperature_c": "inf"},
        ]

        with patch.object(self.server, "HAS_FLOORPLAN_AI", True):
            for payload in invalid_cases:
                with self.subTest(payload=payload):
                    response = self.client.post("/energy/ai_simulate", json=payload)
                    self.assertEqual(response.status_code, 400, response.get_json())

    def test_route_maps_system_specific_default_efficiencies(self):
        heating_defaults = {
            "heat_pump_air": 1.90,
            "heat_pump_geo": 1.90,
            "electric": 1.0,
            "gas_boiler": 0.89,
            "coal_boiler": 0.75,
            "district": 0.80,
            "none": 1.0,
        }
        for system_type, expected in heating_defaults.items():
            with self.subTest(heating=system_type):
                response, params = self.post_and_capture_params({"heating_system": system_type})
                self.assertEqual(response.status_code, 200, response.get_json())
                self.assertEqual(params["heating"]["seasonal_efficiency"], expected)

        cooling_defaults = {
            "central_chiller": 5.0,
            "vrv": 3.8,
            "split_ac": 3.2,
            "evaporative": 8.0,
            "none": 1.0,
        }
        for system_type, expected in cooling_defaults.items():
            with self.subTest(cooling=system_type):
                response, params = self.post_and_capture_params({"cooling_system": system_type})
                self.assertEqual(response.status_code, 200, response.get_json())
                self.assertEqual(params["cooling"]["seasonal_efficiency"], expected)

    def test_route_prefers_public_recovery_percent_and_preserves_manual_efficiency(self):
        response, params = self.post_and_capture_params({
            "heating_system": "gas_boiler",
            "heating_seasonal_efficiency": 0.93,
            "heat_recovery_efficiency": 40,
            "heat_recovery_efficiency_percent": 20,
        })

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(params["heating"]["seasonal_efficiency"], 0.93)
        self.assertEqual(params["ventilation"]["heat_recovery_efficiency"], 0.4)

        response, params = self.post_and_capture_params({
            "heat_recovery_efficiency_percent": 25,
        })
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(params["ventilation"]["heat_recovery_efficiency"], 0.25)

    def test_route_maps_independent_heating_and_cooling_scope(self):
        cases = [
            ({"calculate_heating": True, "calculate_cooling": False}, (True, False)),
            ({"calculate_heating": False, "calculate_cooling": True}, (False, True)),
            ({"calculate_heating": True, "calculate_cooling": True}, (True, True)),
            ({}, (True, True)),
        ]

        for payload, expected in cases:
            with self.subTest(payload=payload):
                response, params = self.post_and_capture_params(payload)
                self.assertEqual(response.status_code, 200, response.get_json())
                self.assertEqual(params["heating"]["enabled"], expected[0])
                self.assertEqual(params["cooling"]["enabled"], expected[1])

    def test_route_rejects_both_calculation_scopes_disabled(self):
        with patch.object(self.server, "HAS_FLOORPLAN_AI", True):
            response = self.client.post(
                "/energy/ai_simulate",
                json={"calculate_heating": False, "calculate_cooling": False},
            )

        self.assertEqual(response.status_code, 400, response.get_json())
        self.assertIn("供暖或制冷", response.get_json()["error"])

    def test_route_rejects_ambiguous_calculation_scope_values(self):
        for value in ("false", 0, 1, None):
            with self.subTest(value=value), patch.object(self.server, "HAS_FLOORPLAN_AI", True):
                response = self.client.post(
                    "/energy/ai_simulate",
                    json={"calculate_heating": value, "calculate_cooling": True},
                )
                self.assertEqual(response.status_code, 400, response.get_json())
                self.assertIn("布尔值", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
