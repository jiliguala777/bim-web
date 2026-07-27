param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentRoot = Join-Path $projectRoot ".venv-train"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"
$requirements = Join-Path $PSScriptRoot "requirements-training.txt"

if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
    & $Python -m venv $environmentRoot
}

& $environmentPython -m pip install --upgrade pip
& $environmentPython -m pip install -r $requirements
& $environmentPython -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

Write-Host "训练环境已准备：$environmentRoot"
