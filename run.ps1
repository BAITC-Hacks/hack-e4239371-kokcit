param([switch]$SkipInstall, [switch]$Check)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

try {
    Push-Location (Join-Path $taskRoot 'backend')
    if (-not $SkipInstall) {
        uv sync --locked
        if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
    }
    if ($Check) {
        & '.\.venv\Scripts\python.exe' -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'Backend tests failed.' }
    }
} finally { Pop-Location }

try {
    Push-Location (Join-Path $taskRoot 'frontend')
    if (-not $SkipInstall) {
        npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    }
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }

if (-not $Check) {
    Write-Host 'WindFlow: http://127.0.0.1:8000  |  Stop: Ctrl+C'
    try {
        Push-Location (Join-Path $taskRoot 'backend')
        & '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8000
        if ($LASTEXITCODE -ne 0) { throw 'Server exited with an error.' }
    } finally { Pop-Location }
}
