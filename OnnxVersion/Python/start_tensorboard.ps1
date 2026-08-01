[CmdletBinding()]
param([int]$Port = 6006)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pathKeys = @(
    [Environment]::GetEnvironmentVariables().Keys |
        Where-Object { $_ -ieq "PATH" }
)
if ($pathKeys.Count -gt 1) {
    [Environment]::SetEnvironmentVariable(
        "PATH", $null, [EnvironmentVariableTarget]::Process
    )
}
$python = Join-Path $projectDirectory "venv\Scripts\python.exe"
$logDirectory = Join-Path $projectDirectory "logs"
$pidDirectory = Join-Path $projectDirectory "pids"
$stdoutPath = Join-Path $logDirectory "tensorboard.stdout.log"
$stderrPath = Join-Path $logDirectory "tensorboard.stderr.log"
$pidPath = Join-Path $pidDirectory "tensorboard.pid"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $pidDirectory | Out-Null

$alreadyRunning = $false
try {
    $existingResponse = Invoke-WebRequest -UseBasicParsing `
        -Uri "http://127.0.0.1:$Port/" -TimeoutSec 2
    $alreadyRunning = $existingResponse.StatusCode -eq 200
}
catch {}
if ($alreadyRunning) {
    Write-Host "TensorBoard is already available: http://localhost:$Port"
    exit 0
}

$process = Start-Process `
    -FilePath $python `
    -ArgumentList @(
        "-m", "tensorboard.main",
        "--logdir", "runs",
        "--host", "127.0.0.1",
        "--port", $Port
    ) `
    -WorkingDirectory $projectDirectory `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id

$deadline = [DateTime]::UtcNow.AddSeconds(15)
do {
    Start-Sleep -Milliseconds 500
    try {
        $response = Invoke-WebRequest -UseBasicParsing `
            -Uri "http://127.0.0.1:$Port/" -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "TensorBoard started (PID $($process.Id)): http://localhost:$Port"
            exit 0
        }
    }
    catch {
        if ($process.HasExited) {
            throw "TensorBoard exited during startup. See $stderrPath"
        }
    }
} while ([DateTime]::UtcNow -lt $deadline)

throw "TensorBoard did not become ready within 15 seconds. See $stderrPath"
