param(
    [string]$DataRoot = "",
    [int]$Port = 8099
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "未找到项目虚拟环境：$python"
}
if ($DataRoot) {
    $env:ANNOTATION_DATA_ROOT = [System.IO.Path]::GetFullPath($DataRoot)
}

Set-Location -LiteralPath $projectRoot
& $python -m annotation_tool.app --host 127.0.0.1 --port $Port
