$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvKaggle = Join-Path $projectRoot '.venv\Scripts\kaggle.exe'
$venvDataCli = Join-Path $projectRoot '.venv\Scripts\credifast-data.exe'
$rawDirectory = Join-Path $projectRoot 'data\raw'
$archivePath = Join-Path $rawDirectory 'home-credit-default-risk.zip'
$credentialFiles = @(
    (Join-Path $env:USERPROFILE '.kaggle\kaggle.json'),
    (Join-Path $env:APPDATA 'kaggle\kaggle.json')
)
$hasCredentialFile = ($credentialFiles | Where-Object {
    Test-Path -LiteralPath $_ -PathType Leaf
}).Count -gt 0
$hasCredentialEnvironment = (
    -not [string]::IsNullOrWhiteSpace($env:KAGGLE_USERNAME) -and
    -not [string]::IsNullOrWhiteSpace($env:KAGGLE_KEY)
)

if (-not (Test-Path -LiteralPath $venvKaggle -PathType Leaf)) {
    throw 'Kaggle CLI is not installed. Run scripts\bootstrap.ps1 first.'
}
if (-not $hasCredentialFile -and -not $hasCredentialEnvironment) {
    throw 'Kaggle credentials are not configured. Configure kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.'
}
if (Test-Path -LiteralPath $rawDirectory -PathType Container) {
    $existingCsv = Get-ChildItem -LiteralPath $rawDirectory -Filter '*.csv' -File
    if ($existingCsv.Count -gt 0) {
        throw "Raw CSV files already exist in $rawDirectory; refusing to overwrite them."
    }
}
else {
    New-Item -ItemType Directory -Path $rawDirectory | Out-Null
}

Push-Location $projectRoot
try {
    & $venvKaggle competitions download -c home-credit-default-risk -p $rawDirectory
    if ($LASTEXITCODE -ne 0) {
        throw 'Kaggle download failed. Confirm credentials and competition-rule acceptance.'
    }
    if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw "Expected archive was not created: $archivePath"
    }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $rawDirectory
    & $venvDataCli manifest --raw-dir $rawDirectory --output artifacts\data_manifest.json
    & $venvDataCli profile-application `
        --input (Join-Path $rawDirectory 'application_train.csv') `
        --output artifacts\application_profile.json
}
finally {
    Pop-Location
}
