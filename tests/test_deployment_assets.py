import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DeploymentAssetTests(unittest.TestCase):
    def read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_gitignore_excludes_runtime_and_large_artifacts(self):
        ignore = self.read(".gitignore")
        for pattern in (
            ".env",
            "models/*.onnx",
            "models/*.pt",
            "tmp/",
            "artifacts/",
            "output/",
            ".playwright-cli/",
            ".codex/",
            ".agents/",
            ".superpowers/",
            "data/*",
        ):
            self.assertIn(pattern, ignore)

    def test_gitignore_keeps_only_runtime_databases(self):
        ignore = self.read(".gitignore")
        for path in (
            "!data/building_library.db",
            "!data/envelope_databases/door.db",
            "!data/envelope_databases/exterior_wall.db",
            "!data/envelope_databases/floor.db",
            "!data/envelope_databases/floor_contact.db",
            "!data/envelope_databases/roof.db",
            "!data/envelope_databases/shading.db",
            "!data/envelope_databases/wall_insulation.db",
            "!data/envelope_databases/window_air_tightness.db",
            "!data/envelope_databases/windows.db",
        ):
            self.assertIn(path, ignore)

    def test_environment_example_has_no_production_defaults(self):
        env = self.read(".env.example")
        for key in (
            "SECRET_KEY",
            "ADMIN_USER",
            "ADMIN_PASSWORD",
            "ONNX_MODEL_PATH",
            "UPLOAD_FOLDER",
            "DB_PATH",
            "BUILDING_LIBRARY_DB_PATH",
            "ENVELOPE_DB_DIR",
        ):
            self.assertRegex(env, rf"(?m)^{key}=")
        self.assertNotIn("M2_DA_best.onnx", env)
        self.assertIn("/opt/bim-web/models/M2_pub_plus_user.onnx", env)

    def test_requirements_support_ubuntu_gunicorn(self):
        requirements = self.read("requirements.txt")
        self.assertRegex(requirements, r"(?m)^gunicorn(?:[<=>].*)?$")
        self.assertRegex(
            requirements,
            r'(?m)^pywin32(?:[<=>].*)?;\s*sys_platform\s*==\s*["\']win32["\']$',
        )

    def test_systemd_service_uses_external_state_and_local_bind(self):
        service = self.read("deploy/bim-web.service")
        for expected in (
            "User=bimweb",
            "WorkingDirectory=/opt/bim-web/app",
            "EnvironmentFile=/etc/bim-web/bim-web.env",
            "/opt/bim-web/venv/bin/gunicorn",
            "--workers 1",
            "--threads 2",
            "--worker-class gthread",
            "--timeout 300",
            "--bind 127.0.0.1:8000",
            "web_server_server:app",
            "Restart=on-failure",
        ):
            self.assertIn(expected, service)

    def test_nginx_proxy_is_ip_compatible_and_does_not_expose_gunicorn(self):
        nginx = self.read("deploy/nginx-bim-web.conf")
        for expected in (
            "listen 80 default_server;",
            "server_name _;",
            "client_max_body_size 200m;",
            "proxy_pass http://127.0.0.1:8000;",
            "proxy_read_timeout 300s;",
            "proxy_send_timeout 300s;",
            "proxy_set_header Host $host;",
            "proxy_set_header X-Real-IP $remote_addr;",
            "proxy_set_header X-Forwarded-Proto $scheme;",
        ):
            self.assertIn(expected, nginx)
        self.assertNotIn("M2_DA_best.onnx", nginx)

    def test_deployment_guide_uses_current_model_and_safe_paths(self):
        guide = self.read("deploy/README.md")
        for expected in (
            "Ubuntu 24.04",
            "libgl1",
            "/opt/bim-web/app",
            "/opt/bim-web/models/M2_pub_plus_user.onnx",
            "/var/lib/bim-web/uploads",
            "/etc/bim-web/bim-web.env",
            "sha256sum",
            "runuser -u bimweb --",
            "不要用 root 直接运行仓库 Git 命令",
            "不要按错误提示给",
            "safe.directory",
            "nginx -t",
            "systemctl status bim-web",
            "journalctl -u bim-web",
            "pull --ff-only origin main",
            "dominant_span_rectangle",
            "房间数量 `19`",
        ):
            self.assertIn(expected, guide)
        self.assertNotIn("Ubuntu 22.04", guide)
        self.assertNotIn("sudo -u bimweb git", guide)
        self.assertNotIn("M2_DA_best.onnx", guide)
        self.assertNotIn(".gemini/antigravity", guide)

    def test_first_upload_checklist_blocks_old_history_and_sensitive_data(self):
        checklist = self.read("docs/GitHub首次上传清单_2026-07-27.md")
        for expected in (
            "Private",
            "不要直接推送当前旧历史",
            "git status --ignored",
            "50 MiB",
            "M2_pub_plus_user.onnx",
            "users.db",
            "data/envelope_databases",
            "git diff --check",
        ):
            self.assertIn(expected, checklist)

    def test_annotation_and_training_operator_assets_are_documented(self):
        for relative_path in (
            "annotation_tool/start_annotation_tool.ps1",
            "training/setup_training_env.ps1",
            "training/requirements-training.txt",
            "annotation_tool/README.md",
        ):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

        annotation_guide = self.read("annotation_tool/README.md")
        for expected in (
            "127.0.0.1",
            r"G:\bim网页\标注数据",
            r"G:\bim网页\标注工具\data",
            "构件骨架粗线",
            "confirmed",
            "single_page_overfit",
        ):
            self.assertIn(expected, annotation_guide)

        project_guide = self.read("README.md")
        for expected in (
            r".\annotation_tool\start_annotation_tool.ps1",
            r".\training\setup_training_env.ps1",
            "-m training.run_experiment",
            "ONNX_MODEL_PATH",
            "single_page_overfit",
            "不自动部署到生产服务器",
        ):
            self.assertIn(expected, project_guide)

        launcher = self.read("annotation_tool/start_annotation_tool.ps1")
        self.assertIn("--host 127.0.0.1", launcher)
        requirements = self.read("training/requirements-training.txt")
        self.assertIn("segmentation-models-pytorch", requirements)
        self.assertIn("onnxruntime", requirements)


if __name__ == "__main__":
    unittest.main()
