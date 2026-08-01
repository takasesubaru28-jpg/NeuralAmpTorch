[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectDirectory "venv\Scripts\python.exe"
$logDirectory = Join-Path $projectDirectory "logs"
$pidDirectory = Join-Path $projectDirectory "pids"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $pidDirectory | Out-Null
Set-Content -LiteralPath (Join-Path $pidDirectory "comparison_runner.pid") -Value $PID

$configurations = @(
    "wavenet_direct_nogan.json",
    "wavenet_direct_gan.json"
)

function Wait-TrainingProcess {
    param(
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [int]$CleanupTimeoutSeconds = 60
    )

    $completionSeenAt = $null
    while (-not $Process.HasExited) {
        if ($null -eq $completionSeenAt -and
            (Test-Path -LiteralPath $StdoutPath) -and
            (Select-String -LiteralPath $StdoutPath -Pattern '^TRAINING_COMPLETED$' -Quiet)) {
            $completionSeenAt = [DateTime]::UtcNow
        }

        if ($null -ne $completionSeenAt -and
            ([DateTime]::UtcNow - $completionSeenAt).TotalSeconds -ge $CleanupTimeoutSeconds) {
            Write-Warning "Training completed but Python did not exit within $CleanupTimeoutSeconds seconds; terminating the stale process."
            Stop-Process -Id $Process.Id -Force
            $Process.WaitForExit()
            return 0
        }
        Start-Sleep -Milliseconds 500
        $Process.Refresh()
    }

    # Read ExitCode before Refresh/disposal. Windows PowerShell can otherwise
    # expose it as $null for redirected child processes.
    return [int]$Process.ExitCode
}

try {
    foreach ($configuration in $configurations) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($configuration)
        $configPath = Join-Path $projectDirectory "configs\$configuration"
        $stdoutPath = Join-Path $logDirectory "$name.stdout.log"
        $stderrPath = Join-Path $logDirectory "$name.stderr.log"
        $pidPath = Join-Path $pidDirectory "$name.pid"

        $process = Start-Process `
            -FilePath $python `
            -ArgumentList @("train.py", "--config", $configPath) `
            -WorkingDirectory $projectDirectory `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -WindowStyle Hidden `
            -PassThru
        Set-Content -LiteralPath $pidPath -Value $process.Id
        $exitCode = Wait-TrainingProcess `
            -Process $process `
            -StdoutPath $stdoutPath
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        if ($exitCode -ne 0) {
            throw "$name failed with exit code $exitCode. See $stderrPath"
        }
    }
}
finally {
    Remove-Item -LiteralPath (Join-Path $pidDirectory "comparison_runner.pid") `
        -Force -ErrorAction SilentlyContinue
}
