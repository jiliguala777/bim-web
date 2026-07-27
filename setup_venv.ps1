$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
$BasePython = $env:BIM_WEB_BASE_PYTHON

function Find-BasePython {
    if ($BasePython -and (Test-Path -LiteralPath $BasePython)) {
        return $BasePython
    }

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        $Candidate = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $Candidate -and (Test-Path -LiteralPath $Candidate)) {
            return $Candidate.Trim()
        }
    }

    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand -and (Test-Path -LiteralPath $PythonCommand.Source)) {
        return $PythonCommand.Source
    }

    throw 'Base Python not found. Install Python 3.12 or set BIM_WEB_BASE_PYTHON to python.exe.'
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $Python $($Arguments -join ' ')"
    }
}

$BasePython = Find-BasePython
Write-Host "Using base Python: $BasePython"

$VenvUsable = $false
if (Test-Path -LiteralPath $VenvPython) {
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $VenvPython -c 'import sys' 2>$null
        $VenvUsable = ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}
if (-not $VenvUsable) {
    if (Test-Path -LiteralPath $VenvRoot) {
        $ResolvedVenvRoot = [System.IO.Path]::GetFullPath($VenvRoot)
        $ResolvedProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
        if (-not $ResolvedVenvRoot.StartsWith("$ResolvedProjectRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove virtual environment outside project root: $ResolvedVenvRoot"
        }
        Remove-Item -LiteralPath $ResolvedVenvRoot -Recurse -Force
    }
    Invoke-Python $BasePython -m venv $VenvRoot
}

Invoke-Python $VenvPython -m pip install --upgrade pip
Invoke-Python $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt') -c (Join-Path $ProjectRoot 'requirements-runtime-constraints.txt')
Invoke-Python $VenvPython -m pip check
