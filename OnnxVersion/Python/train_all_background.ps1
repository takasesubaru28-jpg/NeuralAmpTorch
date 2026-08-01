[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$CleanLogs
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDirectory = Join-Path $projectDirectory "logs"
$pidDirectory = Join-Path $projectDirectory "pids"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvPython = Join-Path $projectDirectory "venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $Python = $venvPython
    }
    else {
        $command = Get-Command python -ErrorAction Stop
        $Python = $command.Source
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable was not found: $Python"
}

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $pidDirectory | Out-Null

if ($CleanLogs) {
    Get-ChildItem -LiteralPath $logDirectory -File |
        Remove-Item -Force
    Get-ChildItem -LiteralPath $pidDirectory -File |
        Remove-Item -Force
}

$configurations = @(
    "gru.json",
    "lstm.json",
    "lru.json",
    "wavenet.json"
)

foreach ($configuration in $configurations) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($configuration)
    $pidPath = Join-Path $pidDirectory "$name.pid"

    if (Test-Path -LiteralPath $pidPath) {
        $existingPid = Get-Content -LiteralPath $pidPath -ErrorAction SilentlyContinue
        $existingProcess = if ($existingPid) {
            Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        }
        $isExpectedPython = $false
        if ($existingProcess) {
            try {
                $isExpectedPython = (
                    [System.IO.Path]::GetFullPath($existingProcess.Path) -eq
                    [System.IO.Path]::GetFullPath($Python)
                )
            }
            catch {
                $isExpectedPython = $existingProcess.ProcessName -match "^python"
            }
        }
        if ($isExpectedPython) {
            throw "$name is already running as PID $existingPid. Wait for it to finish before starting sequential training."
        }
        Remove-Item -LiteralPath $pidPath -Force
        Write-Host "[$name] Removed stale PID file ($existingPid)."
    }
}

$results = @()
foreach ($configuration in $configurations) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($configuration)
    $configPath = Join-Path $projectDirectory "configs\$configuration"
    $stdoutPath = Join-Path $logDirectory "$name.stdout.log"
    $stderrPath = Join-Path $logDirectory "$name.stderr.log"
    $pidPath = Join-Path $pidDirectory "$name.pid"

    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList @("train.py", "--config", $configPath) `
        -WorkingDirectory $projectDirectory `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru

    $process.PriorityClass = "BelowNormal"
    Set-Content -LiteralPath $pidPath -Value $process.Id
    Write-Host "[$name] Training started (PID $($process.Id))."
    Write-Host "[$name] Waiting for training to finish before starting the next model..."

    $process.WaitForExit()
    $process.Refresh()
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue

    $results += [PSCustomObject]@{
        Model = $name
        PID = $process.Id
        ExitCode = $process.ExitCode
        Result = if ($process.ExitCode -eq 0) { "Completed" } else { "Failed" }
        Stdout = $stdoutPath
        Stderr = $stderrPath
    }

    if ($process.ExitCode -eq 0) {
        Write-Host "[$name] Training completed."
    }
    else {
        Write-Warning "[$name] Training failed with exit code $($process.ExitCode). Continuing with the next model."
    }
}

$results | Format-Table -AutoSize
Write-Host "All training processes have finished."
Write-Host "Logs are available in $logDirectory."
