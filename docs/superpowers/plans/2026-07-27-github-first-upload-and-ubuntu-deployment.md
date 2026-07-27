# GitHub First Upload and Ubuntu Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare a safe private-GitHub baseline and reviewable Ubuntu 22.04 deployment templates without uploading models, user data, research data, credentials, or the repository's oversized existing history.

**Architecture:** Keep application code in `/opt/bim-web/app`, the Python environment in `/opt/bim-web/venv`, the externally uploaded ONNX model in `/opt/bim-web/models`, and mutable data under `/var/lib/bim-web`. Nginx exposes port 80 and proxies to a single Gunicorn `gthread` worker bound to localhost; systemd loads production-only settings from `/etc/bim-web/bim-web.env`.

**Tech Stack:** Git ignore rules, Python 3.10, Flask, Gunicorn, systemd, Nginx, ONNX Runtime, Poppler, Python `unittest`.

## Global Constraints

- Target Ubuntu 22.04.
- Initial public access uses the server IP over HTTP; domain and HTTPS are out of scope.
- `models/M2_pub_plus_user.onnx` must not enter Git and must be uploaded separately.
- Do not delete or move local research data, models, uploads, caches, or legacy deployment files.
- Do not run `git add`, `git commit`, `git push`, history rewriting, worktree switching, or cleanup commands.
- Do not expose real IP addresses, passwords, API keys, SSH keys, or server credentials.
- Preserve all unrelated tracked and untracked workspace changes.
- The new deployment baseline must not reference `M2_DA_best.onnx` or `.gemini/antigravity`.

---

### Task 1: Repository Boundary, Environment Example, and Linux Dependencies

**Files:**
- Create: `.env.example`
- Modify: `.gitignore`
- Modify: `requirements.txt`
- Create: `tests/test_deployment_assets.py`

**Interfaces:**
- Consumes: Environment-variable support already present in `web_server_server.py`.
- Produces: Shared ignore rules, a safe environment template, and Ubuntu-installable requirements used by later deployment assets.

- [ ] **Step 1: Write failing repository-boundary tests**

Create `tests/test_deployment_assets.py` with:

```python
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
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_assets -v
```

Expected: failures because `.env.example`, the new ignore patterns, and Gunicorn/Linux markers do not yet exist.

- [ ] **Step 3: Extend `.gitignore` without removing current protections**

Append organized rules that:

```gitignore
# 环境变量与本机配置
.env
.env.*
!.env.example

# 本地工具状态
.playwright-cli/
.codex/
.agents/
.superpowers/

# 临时输出与验证产物
tmp/
artifacts/
output/
.venv-*/

# 模型权重单独交付，不进入 Git
models/*.pt
models/*.pth
models/*.onnx
models/*.ckpt

# SQLite 临时文件和备份
*.db-wal
*.db-shm
*_before_*.db

# 数据目录默认排除，仅保留正式运行数据库
data/*
!data/building_library.db
!data/envelope_databases/
data/envelope_databases/*
!data/envelope_databases/door.db
!data/envelope_databases/exterior_wall.db
!data/envelope_databases/floor.db
!data/envelope_databases/floor_contact.db
!data/envelope_databases/roof.db
!data/envelope_databases/shading.db
!data/envelope_databases/wall_insulation.db
!data/envelope_databases/window_air_tightness.db
!data/envelope_databases/windows.db
```

Retain existing rules for uploads, jobs, logs, `users.db`, private keys, and named sensitive documents.

- [ ] **Step 4: Create the safe environment template**

Create `.env.example`:

```dotenv
SECRET_KEY=replace-with-a-long-random-value
ADMIN_USER=admin
ADMIN_PASSWORD=replace-with-a-strong-password
ONNX_MODEL_PATH=/opt/bim-web/models/M2_pub_plus_user.onnx
UPLOAD_FOLDER=/var/lib/bim-web/uploads
DB_PATH=/var/lib/bim-web/users.db
BUILDING_LIBRARY_DB_PATH=/opt/bim-web/app/data/building_library.db
ENVELOPE_DB_DIR=/opt/bim-web/app/data/envelope_databases
```

- [ ] **Step 5: Make requirements portable to Ubuntu**

Modify `requirements.txt` so it contains:

```text
gunicorn>=22,<24
pywin32; sys_platform == "win32"
```

Keep the remaining dependencies unchanged and do not duplicate existing package lines.

- [ ] **Step 6: Run repository-boundary tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_assets -v
```

Expected: all Task 1 tests pass.

### Task 2: systemd and Nginx Deployment Templates

**Files:**
- Modify: `tests/test_deployment_assets.py`
- Create: `deploy/bim-web.service`
- Create: `deploy/nginx-bim-web.conf`

**Interfaces:**
- Consumes: `.env.example` variable names and the fixed server directory layout.
- Produces: A localhost-only Gunicorn service and an IP-compatible Nginx reverse proxy.

- [ ] **Step 1: Add failing deployment-template tests**

Append these methods to `DeploymentAssetTests`:

```python
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
```

- [ ] **Step 2: Run the deployment-template tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_assets -v
```

Expected: failures because both templates are missing.

- [ ] **Step 3: Create `deploy/bim-web.service`**

Use this exact service structure:

```ini
[Unit]
Description=BIM Web Flask Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=bimweb
Group=bimweb
WorkingDirectory=/opt/bim-web/app
EnvironmentFile=/etc/bim-web/bim-web.env
ExecStart=/opt/bim-web/venv/bin/gunicorn --workers 1 --threads 2 --worker-class gthread --timeout 300 --bind 127.0.0.1:8000 --access-logfile - --error-logfile - web_server_server:app
Restart=on-failure
RestartSec=5
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 4: Create `deploy/nginx-bim-web.conf`**

Use:

```nginx
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    client_max_body_size 200m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_connect_timeout 30s;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

- [ ] **Step 5: Run deployment-template tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_assets -v
```

Expected: all Task 1 and Task 2 tests pass.

### Task 3: Deployment Guide and GitHub First-Upload Checklist

**Files:**
- Modify: `tests/test_deployment_assets.py`
- Create: `deploy/README.md`
- Create: `docs/GitHub首次上传清单_2026-07-27.md`

**Interfaces:**
- Consumes: The environment, systemd, Nginx, Git ignore, model, and directory decisions from Tasks 1 and 2.
- Produces: Human-executable deployment and publication procedures with explicit safety gates.

- [ ] **Step 1: Add failing documentation-content tests**

Append:

```python
    def test_deployment_guide_uses_current_model_and_safe_paths(self):
        guide = self.read("deploy/README.md")
        for expected in (
            "Ubuntu 22.04",
            "/opt/bim-web/app",
            "/opt/bim-web/models/M2_pub_plus_user.onnx",
            "/var/lib/bim-web/uploads",
            "/etc/bim-web/bim-web.env",
            "sha256sum",
            "nginx -t",
            "systemctl status bim-web",
            "journalctl -u bim-web",
            "git pull --ff-only",
        ):
            self.assertIn(expected, guide)
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
```

- [ ] **Step 2: Run tests and verify documentation tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_assets -v
```

Expected: failures because both documents are missing.

- [ ] **Step 3: Write `deploy/README.md`**

Document exact commands for:

```bash
sudo systemctl list-unit-files | grep -Ei 'openclaw|nginx|bim'
sudo ls -la /etc/nginx/sites-enabled
sudo cp -a /etc/nginx/sites-enabled "/etc/nginx/sites-enabled.backup-$(date +%Y%m%d-%H%M%S)"
sudo apt update
sudo apt install -y python3-venv python3-pip nginx poppler-utils libgomp1 git
sudo useradd --system --create-home --home-dir /opt/bim-web --shell /usr/sbin/nologin bimweb
sudo install -d -o bimweb -g bimweb /opt/bim-web/app /opt/bim-web/models /opt/bim-web/venv
sudo install -d -o bimweb -g bimweb /var/lib/bim-web/uploads
sudo install -d -o root -g bimweb -m 0750 /etc/bim-web
sudo -u bimweb git clone PRIVATE_REPOSITORY_URL /opt/bim-web/app
sudo -u bimweb python3 -m venv /opt/bim-web/venv
sudo -u bimweb /opt/bim-web/venv/bin/pip install --upgrade pip
sudo -u bimweb /opt/bim-web/venv/bin/pip install -r /opt/bim-web/app/requirements.txt -c /opt/bim-web/app/requirements-runtime-constraints.txt
sha256sum /opt/bim-web/models/M2_pub_plus_user.onnx
sudo cp /opt/bim-web/app/.env.example /etc/bim-web/bim-web.env
sudo chmod 0640 /etc/bim-web/bim-web.env
sudo chown root:bimweb /etc/bim-web/bim-web.env
sudo cp /opt/bim-web/app/deploy/bim-web.service /etc/systemd/system/bim-web.service
sudo cp /opt/bim-web/app/deploy/nginx-bim-web.conf /etc/nginx/sites-available/bim-web
sudo ln -s /etc/nginx/sites-available/bim-web /etc/nginx/sites-enabled/bim-web
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now bim-web
sudo systemctl reload nginx
curl -I http://127.0.0.1:8000/login
curl -I http://127.0.0.1/login
sudo systemctl status bim-web --no-pager
sudo journalctl -u bim-web -n 100 --no-pager
```

The guide must explicitly instruct the operator to replace `PRIVATE_REPOSITORY_URL`, edit every placeholder in `/etc/bim-web/bim-web.env`, upload the model separately, compare its SHA-256 with the local file, review legacy OpenClaw sites before disabling them, and never expose port 8000 in the Tencent Cloud firewall.

Document updates with:

```bash
cd /opt/bim-web/app
sudo -u bimweb git pull --ff-only
sudo -u bimweb /opt/bim-web/venv/bin/pip install -r requirements.txt -c requirements-runtime-constraints.txt
sudo systemctl restart bim-web
sudo systemctl status bim-web --no-pager
```

Document rollback by checking out a known prior commit only after preserving mutable data outside the code directory.

- [ ] **Step 4: Write the first-upload checklist**

Create `docs/GitHub首次上传清单_2026-07-27.md` with these checked sections:

1. Create a GitHub repository with visibility `Private`.
2. Do not directly push the current old history because it contains oversized model/data objects.
3. Build a clean root commit containing only the approved source, tests, templates, docs, runtime DB files, and deployment assets.
4. Verify ignored files with `git status --ignored`.
5. Scan selected ordinary Git files and reject anything above `50 MiB`.
6. Scan selected content for private-key markers, common secret assignments, and real server addresses without printing secret values.
7. Confirm every model format, `users.db`, uploads, research data, backups, logs, caches, and `.env` is absent.
8. Confirm the ten approved runtime databases under `data/` remain present.
9. Run the full test suite, `py_compile`, inline JavaScript syntax checking, and `git diff --check`.
10. Review GitHub's file list after the first push.
11. Clone on the server, upload `M2_pub_plus_user.onnx` separately, and create the production environment file.

Include a literal approved-file inventory and commands that are read-only or validation-only. State that actual clean-history construction and GitHub push are separate authorized actions.

- [ ] **Step 5: Run documentation tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_assets -v
```

Expected: all deployment-asset tests pass.

### Task 4: Full Verification and Handoff

**Files:**
- Verify: `.gitignore`
- Verify: `.env.example`
- Verify: `requirements.txt`
- Verify: `deploy/bim-web.service`
- Verify: `deploy/nginx-bim-web.conf`
- Verify: `deploy/README.md`
- Verify: `docs/GitHub首次上传清单_2026-07-27.md`
- Verify: `tests/test_deployment_assets.py`

**Interfaces:**
- Consumes: All prior task outputs.
- Produces: Evidence that the workspace is safer to publish and the templates are internally consistent.

- [ ] **Step 1: Verify ignore behavior**

Run:

```powershell
@(
  '.env',
  'models/M2_pub_plus_user.onnx',
  'models/M2_pub_plus_user.pt',
  'uploads/example.pdf',
  'tmp/example.txt',
  'artifacts/example.png',
  'data/envelope_thermal_full/example.pdf',
  'data/envelope_databases/windows_before_classified_thermal_hits_20260708.db'
) | git -c safe.directory=G:/bim-web check-ignore -v --stdin
```

Expected: every listed path is ignored.

Run:

```powershell
@(
  'data/building_library.db',
  'data/envelope_databases/door.db',
  'data/envelope_databases/exterior_wall.db',
  'data/envelope_databases/floor.db',
  'data/envelope_databases/floor_contact.db',
  'data/envelope_databases/roof.db',
  'data/envelope_databases/shading.db',
  'data/envelope_databases/wall_insulation.db',
  'data/envelope_databases/window_air_tightness.db',
  'data/envelope_databases/windows.db'
) | ForEach-Object {
  if (git -c safe.directory=G:/bim-web check-ignore -q -- $_) {
    throw "Runtime database is unexpectedly ignored: $_"
  }
}
```

Expected: no exception.

- [ ] **Step 2: Run focused and full Python tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_assets -v
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: both commands complete with `OK`.

- [ ] **Step 3: Compile changed Python sources**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile web_server_server.py energy_calc.py design_load_calc.py energy_library.py floorplan_onnx.py floorplan_ocr.py floorplan_rooms.py floorplan_topology_repair.py vector_pdf_scale.py tests/test_deployment_assets.py
```

Expected: exit code 0.

- [ ] **Step 4: Validate deployment assets statically**

Run:

```powershell
rg -n "M2_DA_best|\.gemini/antigravity|PRIVATE KEY" .env.example deploy/bim-web.service deploy/nginx-bim-web.conf deploy/README.md docs/GitHub首次上传清单_2026-07-27.md
```

Expected: no matches.

Run:

```powershell
git -c safe.directory=G:/bim-web diff --check
```

Expected: no new whitespace errors. Existing line-ending warnings may remain unchanged.

- [ ] **Step 5: Report the final scope**

Report:

- Which files were created or modified.
- That no data, model, Git history, commit, remote, or server was changed.
- Focused and full test results with exact counts.
- Any pre-existing failures or warnings.
- That actual GitHub creation/push and real server deployment remain separate future actions.
