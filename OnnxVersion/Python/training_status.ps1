$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidDirectory = Join-Path $projectDirectory "pids"

if (-not (Test-Path -LiteralPath $pidDirectory)) {
    Write-Host "No training PID directory exists."
    exit 0
}

$rows = foreach ($pidFile in Get-ChildItem -LiteralPath $pidDirectory -Filter "*.pid") {
    $processId = Get-Content -LiteralPath $pidFile.FullName -ErrorAction SilentlyContinue
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    [PSCustomObject]@{
        Model = $pidFile.BaseName
        PID = $processId
        Running = [bool]$process
        CPUSeconds = if ($process) { [math]::Round($process.CPU, 1) } else { $null }
        Started = if ($process) { $process.StartTime } else { $null }
    }
}

$rows | Format-Table -AutoSize
