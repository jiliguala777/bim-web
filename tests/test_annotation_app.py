import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas

from annotation_tool.app import create_app
from annotation_tool.config import AnnotationConfig
from annotation_tool.services import AnnotationService


def _two_page_pdf() -> bytes:
    buffer = io.BytesIO()
    document = canvas.Canvas(buffer, pagesize=(240, 160))
    document.rect(20, 20, 100, 80)
    document.showPage()
    document.rect(40, 30, 120, 90)
    document.showPage()
    document.save()
    return buffer.getvalue()


class AnnotationApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        config = AnnotationConfig(data_root=Path(self.temporary.name) / "annotation-data")
        self.app = create_app(config)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary.cleanup()

    def _create_project(self) -> dict:
        response = self.client.post(
            "/api/projects",
            data={
                "name": "Four Floor Building",
                "pdf": (io.BytesIO(_two_page_pdf()), "building.pdf"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()["project"]

    def test_pdf_upload_creates_a_project_and_lists_it(self):
        project = self._create_project()

        self.assertEqual(project["page_count"], 2)
        self.assertEqual(project["source_original_name"], "building.pdf")
        listing = self.client.get("/api/projects")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.get_json()["projects"][0]["project_id"], project["project_id"])

    def test_prepare_rejects_page_outside_the_pdf(self):
        project = self._create_project()

        response = self.client.post(
            f"/api/projects/{project['project_id']}/pages/prepare",
            json={"page_number": 3},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("page_number", response.get_json()["error"])

    def test_artifact_route_rejects_names_that_can_escape_the_page(self):
        project = self._create_project()

        response = self.client.get(
            f"/api/projects/{project['project_id']}/pages/1/artifact/%2e%2e%2fproject.json"
        )

        self.assertIn(response.status_code, (400, 404))

    def test_indexed_png_mask_preserves_palette_indices(self):
        indexed = Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8), mode="P")
        palette = [0, 0, 0, 231, 76, 60, 52, 152, 219, 46, 204, 113]
        indexed.putpalette(palette + [0] * (768 - len(palette)))
        buffer = io.BytesIO()
        indexed.save(buffer, format="PNG")

        decoded = AnnotationService.decode_mask(buffer.getvalue())

        self.assertTrue(
            np.array_equal(decoded, np.array([[0, 1], [2, 3]], dtype=np.uint8))
        )

    def test_prepare_and_preannotate_use_the_shared_page_contract(self):
        project = self._create_project()
        service = self.app.extensions["annotation_service"]

        class FakeSegmenter:
            def predict(self, image, **kwargs):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[1:3, 1:3] = 1
                return {"mask": mask}

        service.segmenter_factory = FakeSegmenter

        def fake_prepare(source, page_number, **kwargs):
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            for key, filename in {
                "render": "page_render.png",
                "cleaned": "cleaned_page.png",
                "model_view": "model_view.png",
                "model_input_512": "model_input_512.png",
            }.items():
                self.assertTrue(
                    cv2.imwrite(str(output_dir / filename), np.full((4, 4, 3), 255, np.uint8))
                )
                artifacts[key] = {"path": filename, "sha256": key}
            (output_dir / "preprocessing.json").write_text("{}", encoding="utf-8")
            artifacts["metadata"] = {
                "path": "preprocessing.json",
                "sha256": "metadata",
            }
            return SimpleNamespace(
                page_count=2,
                render_bgr=np.full((4, 4, 3), 255, np.uint8),
                model_input_metadata={
                    "original_size": [4, 4],
                    "resized_size": [512, 512],
                    "padding": [0, 0, 0, 0],
                },
                scale_calibration={},
                vector_cleanup={},
                inference_roi=None,
                artifacts=artifacts,
            )

        with patch("annotation_tool.services.prepare_pdf_page", side_effect=fake_prepare) as prepare:
            response = self.client.post(
                f"/api/projects/{project['project_id']}/pages/prepare",
                json={"page_number": 1},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(prepare.call_args.args[1], 1)
        artifact = self.client.get(
            f"/api/projects/{project['project_id']}/pages/1/artifact/model_view"
        )
        self.assertEqual(artifact.status_code, 200)
        artifact.close()
        preannotation = self.client.post(
            f"/api/projects/{project['project_id']}/pages/1/preannotate"
        )
        self.assertEqual(preannotation.status_code, 200, preannotation.get_json())
        self.assertEqual(preannotation.get_json()["status"], "preannotated")

    def test_draft_confirmation_and_single_page_export_keep_preparation_data(self):
        project = self._create_project()
        store = self.app.extensions["annotation_store"]
        page_dir = store.project_directory(project["project_id"]) / "pages" / "1"
        page_dir.mkdir(parents=True)
        model_view = page_dir / "model_view.png"
        self.assertTrue(cv2.imwrite(str(model_view), np.full((4, 4, 3), 255, np.uint8)))
        model_input = page_dir / "model_input_512.png"
        self.assertTrue(
            cv2.imwrite(str(model_input), np.full((512, 512, 3), 255, np.uint8))
        )
        store.save_page_manifest(
            project["project_id"],
            1,
            {
                "page_count": 2,
                "model_input": {
                    "original_size": [4, 4],
                    "resized_size": [512, 512],
                    "padding": [0, 0, 0, 0],
                },
                "artifacts": {
                    "model_view": {
                        "path": str(model_view.relative_to(store.root)),
                        "sha256": "test",
                    },
                    "model_input_512": {
                        "path": str(model_input.relative_to(store.root)),
                        "sha256": "test-512",
                    }
                },
            },
        )

        response = self.client.post(
            f"/api/projects/{project['project_id']}/pages/1/mask",
            json={
                "mask": [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                "author": "tester",
            },
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["status"], "draft")
        pages = self.client.get(f"/api/projects/{project['project_id']}/pages")
        self.assertEqual(pages.get_json()["pages"][0]["status"], "draft")
        self.assertEqual(store.list_confirmed_pages(project["project_id"]), [])

        confirmation = self.client.post(
            f"/api/projects/{project['project_id']}/pages/1/confirm",
            json={"author": "tester"},
        )
        self.assertEqual(confirmation.status_code, 200, confirmation.get_json())
        self.assertEqual(confirmation.get_json()["status"], "confirmed")
        exported = self.client.post(f"/api/projects/{project['project_id']}/export")
        self.assertEqual(exported.status_code, 201, exported.get_json())
        self.assertEqual(
            exported.get_json()["export"]["experiment_type"],
            "single_page_overfit",
        )
        export_id = exported.get_json()["export"]["export_id"]
        export_root = store.root / "exports" / export_id
        manifest = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
        exported_image = cv2.imread(
            str(export_root / manifest["samples"][0]["image"]),
            cv2.IMREAD_COLOR,
        )
        exported_mask = np.load(
            export_root / manifest["samples"][0]["mask"],
            allow_pickle=False,
        )
        self.assertEqual(exported_image.shape[:2], (512, 512))
        self.assertEqual(exported_mask.shape, (512, 512))
        self.assertEqual(manifest["samples"][0]["status"], "confirmed")


class AnnotationTemplateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        config = AnnotationConfig(data_root=Path(self.temporary.name) / "annotation-data")
        self.app = create_app(config)
        self.app.config["TESTING"] = True

    def tearDown(self):
        self.temporary.cleanup()

    def test_workspace_contains_the_annotation_controls(self):
        response = self.app.test_client().get("/")
        markup = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for element_id in (
            "pdf-file",
            "page-select",
            "view-original",
            "view-cleaned",
            "view-model",
            "mask-canvas",
            "undo",
            "redo",
            "save",
            "confirm",
            "preannotate",
            "export",
        ):
            self.assertIn(f'id="{element_id}"', markup)
        for class_id in (1, 2, 3):
            self.assertIn(f'data-class-id="{class_id}"', markup)

    def test_default_segmenter_uses_the_configured_onnx_model_path(self):
        store = self.app.extensions["annotation_store"]
        with patch.dict(os.environ, {"ONNX_MODEL_PATH": r"G:\models\custom.onnx"}):
            with patch("annotation_tool.services.get_segmenter") as get_segmenter:
                service = AnnotationService(store)
                service._segmenter()

        get_segmenter.assert_called_once_with(r"G:\models\custom.onnx")


if __name__ == "__main__":
    unittest.main()
