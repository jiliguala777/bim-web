import json
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from reportlab.pdfgen import canvas

import floorplan_page_pipeline as page_pipeline
from floorplan_onnx import FloorplanSegmenterONNX
from floorplan_page_pipeline import (
    VectorAnalysisOutcome,
    _run_vector_analysis,
    _write_image,
    prepare_pdf_page,
)


def _make_two_page_pdf(directory):
    pdf_path = Path(directory) / "two-pages.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(400, 300))
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(20, 280, "FIRST_PAGE_MARKER")
    pdf.rect(25, 100, 80, 80, fill=1, stroke=0)
    pdf.showPage()
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(20, 280, "SECOND_PAGE_MARKER")
    pdf.rect(295, 100, 80, 80, fill=1, stroke=0)
    pdf.save()
    return pdf_path


class _TerminationEscalationProcess:
    def __init__(self):
        self.events = []
        self.returncode = None

    def terminate(self):
        self.events.append("terminate")

    def kill(self):
        self.events.append("kill")

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        if len([event for event in self.events if isinstance(event, tuple)]) == 1:
            raise subprocess.TimeoutExpired("vector-worker", timeout)
        self.returncode = -9
        return self.returncode


class VectorAnalysisWorkerTests(unittest.TestCase):
    @staticmethod
    def _write_worker(directory: str, name: str, body: str) -> Path:
        worker = Path(directory) / name
        worker.write_text(body, encoding="utf-8")
        return worker

    def _run_and_assert_result_directory_cleanup(self, *args, **kwargs):
        created_directories = []
        real_temporary_directory = tempfile.TemporaryDirectory

        def recording_temporary_directory(*factory_args, **factory_kwargs):
            context = real_temporary_directory(*factory_args, **factory_kwargs)
            created_directories.append(Path(context.name))
            return context

        with patch.object(
            page_pipeline.tempfile,
            "TemporaryDirectory",
            side_effect=recording_temporary_directory,
        ):
            outcome = _run_vector_analysis(*args, **kwargs)

        self.assertEqual(len(created_directories), 1)
        self.assertFalse(created_directories[0].exists())
        return outcome

    def test_successful_worker_json_uses_current_python_and_explicit_script(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self._write_worker(
                directory,
                "successful_worker.py",
                """
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "ok": True,
            "page_data": {
                "executable": sys.executable,
                "worker_script": str(Path(__file__).resolve()),
            },
        }
    ),
    encoding="utf-8",
)
""".lstrip(),
            )

            outcome = self._run_and_assert_result_directory_cleanup(
                Path(__file__),
                page_index=0,
                dpi=100,
                timeout_seconds=5,
                worker_script=worker,
            )

        self.assertEqual(outcome.evidence["status"], "completed")
        self.assertEqual(
            Path(outcome.page_data["executable"]).resolve(),
            Path(sys.executable).resolve(),
        )
        self.assertEqual(
            Path(outcome.page_data["worker_script"]).resolve(),
            worker.resolve(),
        )

    def test_production_worker_extracts_only_the_selected_page(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _make_two_page_pdf(directory)

            outcome = self._run_and_assert_result_directory_cleanup(
                source,
                page_index=1,
                dpi=100,
                timeout_seconds=5,
            )

        self.assertEqual(outcome.evidence["status"], "completed")
        extracted_text = {
            span["text"] for span in outcome.page_data["text_spans"]
        }
        self.assertIn("SECOND_PAGE_MARKER", extracted_text)
        self.assertNotIn("FIRST_PAGE_MARKER", extracted_text)

    def test_dedicated_worker_catches_extraction_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            outcome = self._run_and_assert_result_directory_cleanup(
                _make_two_page_pdf(directory),
                page_index=2,
                dpi=100,
                timeout_seconds=5,
            )

        self.assertIsNone(outcome.page_data)
        self.assertEqual(outcome.evidence["status"], "failed")
        self.assertEqual(outcome.evidence["mode"], "raster_fallback")
        self.assertEqual(
            outcome.evidence["reason"],
            "IndexError: page_index is outside the PDF page range",
        )

    def test_abrupt_worker_exit_without_result_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self._write_worker(
                directory,
                "abrupt_worker.py",
                "import os\nos._exit(7)\n",
            )

            outcome = self._run_and_assert_result_directory_cleanup(
                Path(__file__),
                page_index=0,
                dpi=100,
                timeout_seconds=5,
                worker_script=worker,
            )

        self.assertIsNone(outcome.page_data)
        self.assertEqual(outcome.evidence["status"], "failed")
        self.assertIn("without a result", outcome.evidence["reason"])
        self.assertIn("7", outcome.evidence["reason"])

    def test_corrupt_worker_result_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self._write_worker(
                directory,
                "corrupt_worker.py",
                """
from pathlib import Path
import sys

Path(sys.argv[1]).write_text("{not-json", encoding="utf-8")
""".lstrip(),
            )

            outcome = self._run_and_assert_result_directory_cleanup(
                Path(__file__),
                page_index=0,
                dpi=100,
                timeout_seconds=5,
                worker_script=worker,
            )

        self.assertIsNone(outcome.page_data)
        self.assertEqual(outcome.evidence["status"], "failed")
        self.assertIn("invalid vector worker result", outcome.evidence["reason"])

    def test_vector_worker_is_terminated_and_reaped_after_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self._write_worker(
                directory,
                "slow_worker.py",
                "import time\ntime.sleep(30)\n",
            )
            started = time.perf_counter()
            started_processes = []
            popen_calls = []
            real_popen = subprocess.Popen

            def recording_popen(command, *args, **kwargs):
                popen_calls.append((command, kwargs.copy()))
                process = real_popen(command, *args, **kwargs)
                started_processes.append(process)
                return process

            with patch.object(
                page_pipeline.subprocess,
                "Popen",
                side_effect=recording_popen,
            ):
                outcome = self._run_and_assert_result_directory_cleanup(
                    Path(__file__),
                    page_index=0,
                    dpi=100,
                    timeout_seconds=0.05,
                    worker_script=worker,
                )

        self.assertLess(time.perf_counter() - started, 5)
        self.assertIsNone(outcome.page_data)
        self.assertEqual(outcome.evidence["status"], "timed_out")
        self.assertEqual(outcome.evidence["mode"], "raster_fallback")
        self.assertEqual(outcome.evidence["timeout_seconds"], 0.05)
        self.assertEqual(len(started_processes), 1)
        self.assertIsNotNone(started_processes[0].returncode)
        if os.name == "nt":
            self.assertTrue(started_processes[0]._handle.closed)
        self.assertEqual(len(popen_calls), 1)
        command, popen_options = popen_calls[0]
        self.assertIsInstance(command, list)
        self.assertEqual(Path(command[0]).resolve(), Path(sys.executable).resolve())
        self.assertEqual(Path(command[1]).resolve(), worker.resolve())
        self.assertIs(popen_options["shell"], False)

    def test_termination_escalates_to_kill_with_bounded_waits(self):
        process = _TerminationEscalationProcess()

        page_pipeline._terminate_and_reap(
            process,
            terminate_timeout_seconds=0.01,
            kill_timeout_seconds=0.02,
        )

        self.assertEqual(
            process.events,
            ["terminate", ("wait", 0.01), "kill", ("wait", 0.02)],
        )
        self.assertEqual(process.returncode, -9)


class ModelInputPreparationTests(unittest.TestCase):
    def test_prepare_model_input_preserves_aspect_ratio_and_reports_padding(self):
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        image = np.zeros((200, 400, 3), dtype=np.uint8)

        model_view, tensor, metadata = segmenter.prepare_model_input(image)

        self.assertEqual(model_view.shape, (200, 400, 3))
        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(metadata["original_size"], [400, 200])
        self.assertEqual(metadata["resized_size"], [512, 256])
        self.assertEqual(metadata["padding"], [128, 0, 128, 0])
        self.assertAlmostEqual(metadata["resize_scale"], 1.28)


class PdfPagePipelineTests(unittest.TestCase):
    @staticmethod
    def _segmenter():
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        return segmenter

    @staticmethod
    def _make_two_page_pdf(directory):
        return _make_two_page_pdf(directory)

    def test_write_image_supports_unicode_output_directory(self):
        image = np.array(
            [
                [[0, 0, 0], [255, 255, 255]],
                [[0, 0, 255], [0, 255, 0]],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "标注数据" / "页面渲染.png"

            artifact = _write_image(output, image)

            encoded = output.read_bytes()
            decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(decoded, image))
            self.assertEqual(artifact["path"], "页面渲染.png")
            self.assertEqual(artifact["sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertFalse(output.with_name(output.name + ".tmp.png").exists())

    def test_prepare_pdf_page_renders_only_the_requested_one_based_page(self):
        poppler_path = os.environ.get("POPPLER_PATH")
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                2,
                poppler_path=poppler_path,
                segmenter=self._segmenter(),
                output_dir=Path(directory) / "artifacts",
            )

            self.assertEqual(prepared.page_number, 2)
            self.assertEqual(prepared.page_count, 2)
            self.assertEqual(prepared.model_input_512.shape, (1, 3, 512, 512))
            gray = prepared.render_bgr.mean(axis=2)
            midpoint = gray.shape[1] // 2
            self.assertLess(gray[:, midpoint:].mean(), gray[:, :midpoint].mean())
            self.assertIn("render", prepared.artifacts)
            self.assertEqual(len(prepared.artifacts["render"]["sha256"]), 64)

    @patch("floorplan_page_pipeline._run_vector_analysis")
    def test_prepare_pdf_page_allows_30_seconds_for_vector_analysis_by_default(
        self, run_analysis
    ):
        def complete_only_with_30_seconds(source, page_index, dpi, timeout_seconds):
            if timeout_seconds < 30.0:
                return VectorAnalysisOutcome(
                    None,
                    {
                        "status": "timed_out",
                        "mode": "raster_fallback",
                        "timeout_seconds": timeout_seconds,
                        "reason": "vector extraction timed out",
                    },
                )
            return VectorAnalysisOutcome(
                {
                    "page_size_pt": [400.0, 300.0],
                    "render_size_px": [556, 417],
                    "dpi": 100,
                    "text_spans": [],
                    "segments": [],
                    "styled_edges": [],
                    "vector_text_count": 0,
                    "vector_segment_count": 0,
                    "styled_edge_count": 0,
                    "has_vector_text": False,
                    "has_vector_geometry": False,
                    "is_vector_pdf": False,
                },
                {
                    "status": "completed",
                    "mode": "vector",
                    "timeout_seconds": timeout_seconds,
                    "reason": None,
                },
            )

        run_analysis.side_effect = complete_only_with_30_seconds
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                1,
                poppler_path=os.environ.get("POPPLER_PATH"),
                segmenter=self._segmenter(),
            )

        self.assertEqual(prepared.vector_analysis["status"], "completed")

    @patch("floorplan_page_pipeline._run_vector_analysis")
    def test_prepare_pdf_page_writes_raster_fallback_metadata(self, run_analysis):
        run_analysis.return_value = VectorAnalysisOutcome(
            None,
            {
                "status": "timed_out",
                "mode": "raster_fallback",
                "timeout_seconds": 15,
                "reason": "vector extraction exceeded 15 seconds",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                1,
                poppler_path=os.environ.get("POPPLER_PATH"),
                segmenter=self._segmenter(),
                output_dir=output,
            )
            metadata = json.loads(
                (output / "preprocessing.json").read_text(encoding="utf-8")
            )

        self.assertEqual(prepared.vector_analysis["mode"], "raster_fallback")
        self.assertEqual(metadata["render_dpi"], 100)
        self.assertEqual(metadata["vector_analysis"]["status"], "timed_out")
        self.assertEqual(prepared.scale_calibration["status"], "manual_required")
        self.assertEqual(
            prepared.scale_calibration["method"],
            "raster_fallback_manual",
        )
        self.assertFalse(prepared.vector_cleanup["enabled"])
        self.assertIsNone(prepared.inference_roi)
        self.assertIsNone(prepared.structural_support_mask)
        self.assertEqual(
            set(prepared.artifacts),
            {"render", "cleaned", "model_view", "model_input_512", "metadata"},
        )

    @patch("floorplan_page_pipeline._run_vector_analysis")
    def test_prepare_pdf_page_keeps_completed_vector_analysis(self, run_analysis):
        run_analysis.return_value = VectorAnalysisOutcome(
            {
                "page_size_pt": [400.0, 300.0],
                "render_size_px": [556, 417],
                "dpi": 100,
                "text_spans": [],
                "segments": [],
                "styled_edges": [],
                "vector_text_count": 0,
                "vector_segment_count": 0,
                "styled_edge_count": 0,
                "has_vector_text": False,
                "has_vector_geometry": False,
                "is_vector_pdf": False,
            },
            {
                "status": "completed",
                "mode": "vector",
                "timeout_seconds": 15,
                "reason": None,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                1,
                poppler_path=os.environ.get("POPPLER_PATH"),
                segmenter=self._segmenter(),
            )
        self.assertEqual(prepared.vector_analysis["status"], "completed")

    def test_real_worker_timeout_flows_to_100_dpi_fallback_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"

            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                1,
                poppler_path=os.environ.get("POPPLER_PATH"),
                segmenter=self._segmenter(),
                output_dir=output,
                vector_timeout_seconds=0.001,
            )
            metadata = json.loads(
                (output / "preprocessing.json").read_text(encoding="utf-8")
            )

            self.assertEqual(prepared.vector_analysis["status"], "timed_out")
            self.assertEqual(prepared.vector_analysis["mode"], "raster_fallback")
            self.assertEqual(
                prepared.vector_analysis["timeout_seconds"],
                0.001,
            )
            self.assertEqual(metadata["render_dpi"], 100)
            self.assertEqual(
                metadata["scale_calibration"]["method"],
                "raster_fallback_manual",
            )
            self.assertEqual(
                set(prepared.artifacts),
                {
                    "render",
                    "cleaned",
                    "model_view",
                    "model_input_512",
                    "metadata",
                },
            )
            for artifact in prepared.artifacts.values():
                self.assertTrue((output / artifact["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
