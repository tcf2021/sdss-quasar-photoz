param(
    [ValidateSet("install", "check", "prepare", "smoke", "train", "evaluate", "verify", "all", "bundle")]
    [string]$Stage = "check",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$V1Root = "",
    [string]$Python = "",
    [string]$Run = "",
    [int]$Threads = 8
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $V1Root) { $V1Root = Join-Path $PSScriptRoot "..\sdss-photoz" }
$V1Root = [System.IO.Path]::GetFullPath($V1Root)
if (-not $Python) { $Python = Join-Path $V1Root ".venv-win\Scripts\python.exe" }
if (-not (Test-Path $Python)) { throw "Python not found: $Python. Pass -Python with the existing environment path." }
$Python = (Resolve-Path $Python).Path
if (-not $Run) { $Run = "runs/full_$Device" }
New-Item -ItemType Directory -Force "logs" | Out-Null
$env:PYTHONUTF8 = "1"

function Invoke-PhotoZ {
    param([string[]]$PhotoZArguments)
    $previousPreference = $ErrorActionPreference
    $photoZExitCode = $null
    try {
        # Windows PowerShell 5 can wrap ordinary native stderr as ErrorRecord.
        # Decide native command success from its exit code, including crashes.
        $ErrorActionPreference = "Continue"
        & $Python -u -X faulthandler "$PSScriptRoot\v2.py" @PhotoZArguments 2>&1 |
            Tee-Object -FilePath "logs/windows_$Stage.log" -Append -ErrorAction Stop
        $photoZExitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousPreference }
    if ($null -eq $photoZExitCode -or $photoZExitCode -ne 0) {
        throw "V2 command failed. Completed trials are retained. Read the error above before continuing."
    }
}
if ($Stage -eq "install") {
    & uv pip install --python $Python -r "$PSScriptRoot\requirements.txt"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
    & uv pip check --python $Python
    if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }
}
if ($Stage -eq "check") { Invoke-PhotoZ -PhotoZArguments @("check-env", "--device", $Device) }
if ($Stage -in @("prepare", "all")) {
    Invoke-PhotoZ -PhotoZArguments @("prepare", "--v1-data", "$V1Root\data\processed",
        "--raw", "$V1Root\data\raw\DR16Q_v4.fits", "--output", "data/v2")
}
if ($Stage -eq "smoke") {
    Invoke-PhotoZ -PhotoZArguments @("cv", "--data", "data/v2", "--run", "runs/smoke_$Device",
        "--config", "configs/smoke.json", "--device", $Device, "--threads", "$Threads")
}
if ($Stage -in @("train", "all")) {
    Invoke-PhotoZ -PhotoZArguments @("train", "--data", "data/v2", "--run", $Run,
        "--config", "configs/full.json", "--device", $Device, "--threads", "$Threads")
}
if ($Stage -in @("evaluate", "all")) {
    Invoke-PhotoZ -PhotoZArguments @("evaluate", "--data", "data/v2", "--run", $Run)
}
if ($Stage -in @("verify", "all")) {
    Invoke-PhotoZ -PhotoZArguments @("verify", "--data", "data/v2", "--run", $Run)
}
if ($Stage -eq "bundle") {
    Invoke-PhotoZ -PhotoZArguments @("bundle", "--data", "data/v2", "--run", $Run,
        "--output", "sdss-photoz-v2-results.zip")
}
Write-Host "Stage completed: $Stage"
