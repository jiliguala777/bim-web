"""Local-only Flask API for PDF floorplan annotation."""

from __future__ import annotations

import argparse
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from .config import AnnotationConfig
from .services import AnnotationService, PreparationInProgressError
from .storage import AnnotationStore


LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def _version_payload(version, warnings=None) -> dict:
    return {
        "version_id": version.version_id,
        "status": version.status,
        "author": version.author,
        "created_at": version.created_at,
        "previous_version_id": version.previous_version_id,
        "warnings": list(warnings or []),
    }


def create_app(config: AnnotationConfig | None = None) -> Flask:
    annotation_config = config or AnnotationConfig.from_env()
    store = AnnotationStore(annotation_config)
    service = AnnotationService(store)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.environ.get("ANNOTATION_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024))
    )
    app.extensions["annotation_store"] = store
    app.extensions["annotation_service"] = service

    @app.get("/")
    def workspace():
        return render_template("index.html")

    @app.errorhandler(PreparationInProgressError)
    def preparation_conflict(error):
        return jsonify({"error": str(error)}), 409

    @app.errorhandler(ValueError)
    def invalid_request(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(FileNotFoundError)
    def missing_resource(error):
        return jsonify({"error": str(error)}), 404

    @app.get("/api/projects")
    def list_projects():
        return jsonify({"projects": service.list_projects()})

    @app.post("/api/projects")
    def create_project_route():
        project = service.create_project(
            request.files.get("pdf"),
            request.form.get("name", "").strip() or "Floorplan Project",
        )
        return jsonify({"project": project}), 201

    @app.get("/api/projects/<project_id>/pages")
    def list_pages(project_id):
        return jsonify({"pages": service.list_pages(project_id)})

    @app.post("/api/projects/<project_id>/pages/prepare")
    def prepare_page(project_id):
        payload = request.get_json(silent=True) or {}
        if "page_number" not in payload:
            raise ValueError("page_number is required")
        page = service.prepare_page(project_id, int(payload["page_number"]))
        return jsonify({"page": page})

    @app.get("/api/projects/<project_id>/pages/<int:page>/regions")
    def list_regions(project_id, page):
        return jsonify({"regions": service.list_regions(project_id, page)})

    @app.post("/api/projects/<project_id>/pages/<int:page>/regions")
    def create_region(project_id, page):
        body = request.get_json(silent=True) or {}
        if "name" not in body:
            raise ValueError("region name is required")
        if "crop_bbox_px" not in body:
            raise ValueError("crop_bbox_px is required")
        region = service.create_region(
            project_id,
            page,
            body["name"],
            body["crop_bbox_px"],
        )
        return jsonify({"region": region}), 201

    @app.get(
        "/api/projects/<project_id>/pages/<int:page>/regions/"
        "<region_id>/artifact/<name>"
    )
    def get_region_artifact(project_id, page, region_id, name):
        return send_file(
            service.region_artifact_path(
                project_id,
                page,
                region_id,
                name,
            )
        )

    @app.get(
        "/api/projects/<project_id>/pages/<int:page>/regions/<region_id>/mask"
    )
    def get_region_mask(project_id, page, region_id):
        current = service.current_region_mask(project_id, page, region_id)
        if current is None:
            raise FileNotFoundError("Region does not have an annotation mask")
        mask, version = current
        if request.args.get("space") == "model512":
            mask_path = version.mask_path.with_name(
                f"{version.version_id}.mask_512.npy"
            )
            if not mask_path.is_file():
                raise FileNotFoundError("Model-space mask does not exist")
            mask = np.load(mask_path, allow_pickle=False)
        elif request.args.get("space") not in (None, "", "full"):
            raise ValueError("mask space is invalid")
        encoded, buffer = cv2.imencode(".png", mask)
        if not encoded:
            raise ValueError("mask could not be encoded")
        response = send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype="image/png",
            download_name=f"region-{region_id}-{version.version_id}.png",
        )
        response.headers["X-Annotation-Version"] = version.version_id
        response.headers["X-Annotation-Status"] = version.status
        return response

    @app.post(
        "/api/projects/<project_id>/pages/<int:page>/regions/<region_id>/mask"
    )
    def save_region_mask(project_id, page, region_id):
        if "mask" in request.files:
            payload = request.files["mask"].read()
            author = request.form.get("author", "local-user")
        else:
            body = request.get_json(silent=True) or {}
            if "mask" not in body:
                raise ValueError("mask is required")
            payload = body["mask"]
            author = body.get("author", "local-user")
        version, warnings = service.save_region_mask(
            project_id,
            page,
            region_id,
            payload,
            status="draft",
            author=author,
        )
        return jsonify(_version_payload(version, warnings))

    @app.post(
        "/api/projects/<project_id>/pages/<int:page>/regions/<region_id>/confirm"
    )
    def confirm_region(project_id, page, region_id):
        body = request.get_json(silent=True) or {}
        version = service.confirm_region(
            project_id,
            page,
            region_id,
            author=body.get("author", "local-user"),
        )
        return jsonify(_version_payload(version))

    @app.post(
        "/api/projects/<project_id>/pages/<int:page>/regions/"
        "<region_id>/preannotate"
    )
    def preannotate_region(project_id, page, region_id):
        body = request.get_json(silent=True) or {}
        version, warnings = service.preannotate_region(
            project_id,
            page,
            region_id,
            author=body.get("author", "onnx-preannotation"),
            allow_blank=bool(body.get("allow_blank", False)),
        )
        return jsonify(_version_payload(version, warnings))

    @app.get("/api/projects/<project_id>/pages/<int:page>/artifact/<name>")
    def get_artifact(project_id, page, name):
        return send_file(service.artifact_path(project_id, page, name))

    @app.get("/api/projects/<project_id>/pages/<int:page>/mask")
    def get_mask(project_id, page):
        current = service.current_mask(project_id, page)
        if current is None:
            raise FileNotFoundError("Page does not have an annotation mask")
        mask, version = current
        if request.args.get("space") == "model512":
            mask_path = version.mask_path.with_name(f"{version.version_id}.mask_512.npy")
            if not mask_path.is_file():
                raise FileNotFoundError("Model-space mask does not exist")
            mask = np.load(mask_path, allow_pickle=False)
        elif request.args.get("space") not in (None, "", "full"):
            raise ValueError("mask space is invalid")
        encoded, buffer = cv2.imencode(".png", mask)
        if not encoded:
            raise ValueError("mask could not be encoded")
        response = send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype="image/png",
            download_name=f"page-{page}-{version.version_id}.png",
        )
        response.headers["X-Annotation-Version"] = version.version_id
        response.headers["X-Annotation-Status"] = version.status
        return response

    @app.post("/api/projects/<project_id>/pages/<int:page>/mask")
    def save_mask(project_id, page):
        if "mask" in request.files:
            payload = request.files["mask"].read()
            author = request.form.get("author", "local-user")
        else:
            body = request.get_json(silent=True) or {}
            if "mask" not in body:
                raise ValueError("mask is required")
            payload = body["mask"]
            author = body.get("author", "local-user")
        version, warnings = service.save_mask(
            project_id,
            page,
            payload,
            status="draft",
            author=author,
        )
        return jsonify(_version_payload(version, warnings))

    @app.post("/api/projects/<project_id>/pages/<int:page>/confirm")
    def confirm_page(project_id, page):
        body = request.get_json(silent=True) or {}
        version = service.confirm_page(
            project_id,
            page,
            author=body.get("author", "local-user"),
        )
        return jsonify(_version_payload(version))

    @app.post("/api/projects/<project_id>/pages/<int:page>/preannotate")
    def preannotate_page(project_id, page):
        body = request.get_json(silent=True) or {}
        version, warnings = service.preannotate_page(
            project_id,
            page,
            author=body.get("author", "onnx-preannotation"),
            allow_blank=bool(body.get("allow_blank", False)),
        )
        return jsonify(_version_payload(version, warnings))

    @app.post("/api/projects/<project_id>/export")
    def export_project(project_id):
        return jsonify({"export": service.export_confirmed(project_id)}), 201

    return app


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local PDF floorplan annotation tool")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.host not in LOCAL_HOSTS:
        raise SystemExit("Annotation tool only accepts --host 127.0.0.1 or localhost")
    create_app().run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
