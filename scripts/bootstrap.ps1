$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot '.venv'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        python -m venv $venvPath
    }
    & $pythonPath -m pip install --upgrade pip
    & $pythonPath -m pip install -e '.[api,ui,data,ml,dev]'
    & $pythonPath -m unittest discover -s tests -v
}
finally {
    Pop-Location
}
