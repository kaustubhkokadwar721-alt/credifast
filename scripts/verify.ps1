$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pythonCommand = if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $venvPython
}
else {
    'python'
}
Push-Location $projectRoot
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    $ruffCommand = Join-Path $projectRoot '.venv\Scripts\ruff.exe'
    if (Test-Path -LiteralPath $ruffCommand -PathType Leaf) {
        & $ruffCommand check src tests ui
    }
    & $pythonCommand -m unittest discover -s tests -v
    & $pythonCommand -m compileall -q src tests ui
    & $pythonCommand scripts/verify_artifacts.py
    & $pythonCommand -m credifast examples/applicant.json --policy configs/decision_policy.json
}
finally {
    Pop-Location
}
