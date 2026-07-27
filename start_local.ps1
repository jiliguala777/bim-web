$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'Project virtual environment is missing. Run .\setup_venv.ps1 first.'
}
Push-Location $ProjectRoot
try {
    & $VenvPython (Join-Path $ProjectRoot 'run_local.py')
    if ($LASTEXITCODE -ne 0) {
        throw "Local server exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
