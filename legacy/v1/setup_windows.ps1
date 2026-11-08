param([ValidateSet('cuda','cpu')][string]$Mode = 'cuda')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" } }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv with winget install --id=astral-sh.uv -e, reopen PowerShell, then retry.'
}
uv python install 3.11
Check-Exit
if (-not (Test-Path .venv)) {
    uv venv --python 3.11 --seed .venv
    Check-Exit
}
& .\.venv\Scripts\python.exe -c "import sys; assert sys.version_info[:2] == (3,11), 'Use a separate Python 3.11 .venv'"
Check-Exit
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
Check-Exit
$TorchIndex = if ($Mode -eq 'cuda') {'https://download.pytorch.org/whl/cu128'} else {'https://download.pytorch.org/whl/cpu'}
uv pip install --python .\.venv\Scripts\python.exe torch==2.9.0 --index-url $TorchIndex
Check-Exit
& .\.venv\Scripts\python.exe -m pip check
Check-Exit
& .\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 environment-lock.txt
Check-Exit
& .\.venv\Scripts\python.exe project.py check-env --device $Mode
Check-Exit
Write-Output 'Environment ready. Use .\.venv\Scripts\python.exe for the remaining commands.'
