[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Virtual environment not found. Run .\scripts\bootstrap.ps1 first."
}

$ApiOutLog = Join-Path $ProjectRoot "artifacts\demo-api.out.log"
$ApiErrLog = Join-Path $ProjectRoot "artifacts\demo-api.err.log"
$ApiArguments = @(
    "-m", "uvicorn", "credifast.api:app",
    "--host", "127.0.0.1",
    "--port", $ApiPort.ToString()
)

$ApiProcess = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList $ApiArguments `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $ApiOutLog `
    -RedirectStandardError $ApiErrLog `
    -PassThru

try {
    Write-Host "CrediFast API:       http://127.0.0.1:$ApiPort/docs"
    Write-Host "Reviewer dashboard: http://127.0.0.1:$UiPort"
    Write-Host "Press Ctrl+C to stop both services."
    & $PythonPath -m streamlit run "ui/app.py" --server.port $UiPort
}
finally {
    if ($null -ne $ApiProcess -and -not $ApiProcess.HasExited) {
        Stop-Process -Id $ApiProcess.Id -Force
    }
}
