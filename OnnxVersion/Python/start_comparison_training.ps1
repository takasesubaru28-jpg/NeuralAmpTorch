[CmdletBinding()]
param()

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
$runnerPidPath = Join-Path $projectDirectory "pids\comparison_runner.pid"
if (Test-Path -LiteralPath $runnerPidPath) {
    $runnerPid = Get-Content -LiteralPath $runnerPidPath -ErrorAction SilentlyContinue
    if ($runnerPid -and (Get-Process -Id $runnerPid -ErrorAction SilentlyContinue)) {
        throw "Comparison training is already running as PID $runnerPid."
    }
}

$runner = Start-Process `
    -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $projectDirectory "train_comparison_background.ps1")
    ) `
    -WorkingDirectory $projectDirectory `
    -RedirectStandardOutput (Join-Path $projectDirectory "logs\comparison_runner.stdout.log") `
    -RedirectStandardError (Join-Path $projectDirectory "logs\comparison_runner.stderr.log") `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Comparison training runner started (PID $($runner.Id))."
Write-Host "The non-GAN model runs first; the GAN model starts after convergence."
