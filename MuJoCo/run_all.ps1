$ErrorActionPreference = 'Stop'
$python = 'E:\MuJoCo\runtime\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'run_six_algorithms.py'
& $python $script
if ($LASTEXITCODE -ne 0) {
    throw "MuJoCo simulation failed with exit code $LASTEXITCODE"
}
