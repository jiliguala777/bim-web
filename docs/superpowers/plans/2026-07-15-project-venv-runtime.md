# bim-web Project Virtual Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible project-local Python environment that runs Flask, pvlib and the deployed ONNX floor-plan model without relying on the mixed base Anaconda package set.

**Architecture:** A repository-local `.venv` owns all website runtime dependencies. A constraints file pins the NumPy/OpenCV/ONNX/protobuf compatibility boundary; one PowerShell script creates or refreshes the environment and another always starts the site with `.venv\Scripts\python.exe`.

**Tech Stack:** Windows PowerShell, Python 3.12 `venv`, pip constraints, Flask, NumPy 1.26.4, OpenCV headless 4.10.0.84, ONNX Runtime 1.20.1.

## Global Constraints

- Keep `.venv/` local and ignored by Git.
- Pin NumPy 1.26.4, OpenCV headless 4.10.0.84, ONNX Runtime 1.20.1 and protobuf 4.25.8.
- Do not install PyTorch in the website virtual environment.
- Do not hard-code administrator credentials or secrets.
- Do not stage, commit, push or deploy any change.
- Do not modify model structure or recognition preprocessing.

---

### Task 1: Runtime constraints and environment setup script

**Files:**
- Create: `requirements-runtime-constraints.txt`
- Create: `setup_venv.ps1`

**Interfaces:**
- Consumes: `requirements.txt`, base interpreter `D:\Anaconda\Anaconda3\python.exe`.
- Produces: `.venv\Scripts\python.exe` containing the website dependencies.

- [ ] **Step 1: Create the constraints file**

```text
numpy==1.26.4
opencv-python-headless==4.10.0.84
onnxruntime==1.20.1
protobuf==4.25.8
```

- [ ] **Step 2: Create the idempotent setup script**

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$BasePython = 'D:\Anaconda\Anaconda3\python.exe'

if (-not (Test-Path -LiteralPath $BasePython)) {
    throw "Base Python not found: $BasePython"
}
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $BasePython -m venv (Join-Path $ProjectRoot '.venv')
}
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt') -c (Join-Path $ProjectRoot 'requirements-runtime-constraints.txt')
& $VenvPython -m pip check
```

- [ ] **Step 3: Run the setup script**

Run: `powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1`

Expected: exit code 0; `.venv\Scripts\python.exe` exists; `pip check` prints `No broken requirements found.`

- [ ] **Step 4: Confirm pinned versions**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import numpy,cv2,onnxruntime,google.protobuf; print(numpy.__version__,cv2.__version__,onnxruntime.__version__,google.protobuf.__version__)"
```

Expected: `1.26.4 4.10.0 1.20.1 4.25.8`.

---

### Task 2: Dedicated local launcher

**Files:**
- Create: `start_local.ps1`

**Interfaces:**
- Consumes: `.venv\Scripts\python.exe`, `run_local.py`, optional existing `ADMIN_USER`, `ADMIN_PASSWORD`, `SECRET_KEY` and `PORT` environment variables.
- Produces: Flask development server at `http://127.0.0.1:$PORT` without changing credentials.

- [ ] **Step 1: Create the launcher**

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'Project virtual environment is missing. Run .\setup_venv.ps1 first.'
}
Push-Location $ProjectRoot
try {
    & $VenvPython (Join-Path $ProjectRoot 'run_local.py')
} finally {
    Pop-Location
}
```

- [ ] **Step 2: Verify the missing-environment error path independently**

Run a source inspection asserting the launcher contains the exact `.venv` existence check and setup instruction; do not rename the working `.venv` merely to test an error message.

Run:

```powershell
Select-String -Path .\start_local.ps1 -Pattern "Test-Path.*VenvPython","setup_venv.ps1"
```

Expected: both patterns are present.

---

### Task 3: Restore the base environment boundary

**Files:**
- No repository files.

**Interfaces:**
- Consumes: `D:\Anaconda\Anaconda3\python.exe` base environment.
- Produces: protobuf 7.34.1 restored in the base environment; the site no longer depends on its package resolution.

- [ ] **Step 1: Restore the protobuf version that existed before this repair**

Run:

```powershell
D:\Anaconda\Anaconda3\python.exe -m pip install protobuf==7.34.1
```

Expected: protobuf 7.34.1 installed. A remaining Streamlit 1.32 compatibility warning is recorded as a pre-existing base-environment conflict and is outside the website runtime.

- [ ] **Step 2: Prove the project environment is independent**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import google.protobuf; print(google.protobuf.__version__)"
D:\Anaconda\Anaconda3\python.exe -c "import google.protobuf; print(google.protobuf.__version__)"
```

Expected: project prints `4.25.8`; base prints `7.34.1`.

---

### Task 4: Model, application and regression verification

**Files:**
- No production-code changes expected.
- Read: `floorplan_onnx.py`, `models/M2_pub_plus_user.onnx`, `web_server_server.py`.

**Interfaces:**
- Consumes: completed `.venv` and the real deployment model.
- Produces: evidence that the actual website runtime imports, loads and executes the correct components.

- [ ] **Step 1: Verify the complete import chain**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import numpy,cv2,pandas,pyarrow,pvlib,onnxruntime; print('IMPORTS_OK')"
```

Expected: `IMPORTS_OK` and no NumPy 1.x/2.x binary warning.

- [ ] **Step 2: Load and execute the actual ONNX model**

Run a Python probe using `floorplan_onnx.get_segmenter('models/M2_pub_plus_user.onnx')`, obtain the model input shape, construct one zero-valued `uint8` image with the corresponding spatial dimensions, and call `predict(..., use_preprocessing=True)`.

Expected: the session uses `CPUExecutionProvider`; the result contains `mask` and `geometry`; no ONNX Runtime import or model-load error occurs.

- [ ] **Step 3: Run the full automated suite**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests`

Expected: all current tests pass with zero failures.

- [ ] **Step 4: Verify application imports the correct energy engine**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import os,web_server_server as w; print(os.path.abspath(w.energy_calc.__file__)); print(w.HAS_FLOORPLAN_AI,w.HAS_ENERGY_CALC)"
```

Expected: path is `E:\bim-web\energy_calc.py`; both flags are `True`; no missing ONNX Runtime or pvlib error appears.

- [ ] **Step 5: Start and probe the Flask server**

Run `powershell -ExecutionPolicy Bypass -File .\start_local.ps1` in a background terminal, request `http://127.0.0.1:5000/login`, and then stop the test server.

Expected: HTTP 200 and startup output contains none of `numpy.core.multiarray failed to import`, `onnxruntime is required`, or `FloorPlan AI not available`.

- [ ] **Step 6: Check working-tree scope**

Run: `git status --short`

Expected: `.venv` is absent because it is ignored; only the constraints, setup script, launcher and already-authorized project changes appear. Do not stage or commit.
