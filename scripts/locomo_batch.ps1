[CmdletBinding()]
param(
    [string]$SampleIndices = "",
    [int]$SampleCount = 10,
    [int]$BatchSize = 2,
    [string]$DataPath = "data/locomo10.json",
    [string]$OutputRoot = "result/locomo_full_rerun",
    [string]$PythonPath = "",
    [int]$Threads = 2,
    [switch]$Worker,
    [int]$SampleIndex = -1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

function Resolve-RepoPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

$DataPath = Resolve-RepoPath $DataPath
$OutputRoot = Resolve-RepoPath $OutputRoot
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    if (-not [string]::IsNullOrWhiteSpace($env:BENCHMARK_PYTHON)) {
        $PythonPath = $env:BENCHMARK_PYTHON
    } else {
        $PythonPath = (Get-Command python -ErrorAction Stop).Source
    }
}

$PythonPath = Resolve-RepoPath $PythonPath
$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:PYTHONUNBUFFERED = "1"

function Invoke-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$StdOutPath,
        [string]$StdErrPath
    )

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $StdOutPath `
        -RedirectStandardError $StdErrPath `
        -PassThru `
        -Wait
    return $process.ExitCode
}

function Write-WorkerStatus {
    param(
        [string]$Path,
        [int]$Index,
        [string]$Status,
        [int]$ExitCode,
        [string]$Message
    )

    $payload = [ordered]@{
        sample_index = $Index
        status = $Status
        exit_code = $ExitCode
        message = $Message
        updated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $Path -Encoding UTF8
}

if ($Worker) {
    if ($SampleIndex -lt 0 -or $SampleIndex -ge $SampleCount) {
        throw "SampleIndex must be between 0 and SampleCount - 1."
    }

    $sampleDir = Join-Path $OutputRoot ("sample_{0:D2}" -f $SampleIndex)
    New-Item -ItemType Directory -Path $sampleDir -Force | Out-Null
    $statusPath = Join-Path $sampleDir "worker_status.json"

    $ingestArgs = @(
        "-m", "neuramem_benchmark.ingest",
        "--input", $DataPath,
        "--sample", $SampleIndex,
        "--usage-output-dir", $sampleDir
    )
    $ingestCode = Invoke-LoggedProcess `
        -FilePath $PythonPath `
        -ArgumentList $ingestArgs `
        -StdOutPath (Join-Path $sampleDir "ingest.stdout.log") `
        -StdErrPath (Join-Path $sampleDir "ingest.stderr.log")

    $manifestPath = Join-Path $sampleDir ("ingest_manifest_{0}.json" -f $SampleIndex)
    if ($ingestCode -ne 0 -or -not (Test-Path -LiteralPath $manifestPath)) {
        $message = "Ingest failed or manifest missing."
        Write-WorkerStatus $statusPath $SampleIndex "failed" $ingestCode $message
        exit 10
    }

    $evalPath = Join-Path $sampleDir "eval.csv"
    $evalArgs = @(
        "-m", "neuramem_benchmark.runner",
        "--input", $DataPath,
        "--sample", $SampleIndex,
        "--output", $evalPath,
        "--threads", $Threads,
        "--manifest-dir", $sampleDir
    )
    $evalCode = Invoke-LoggedProcess `
        -FilePath $PythonPath `
        -ArgumentList $evalArgs `
        -StdOutPath (Join-Path $sampleDir "eval.stdout.log") `
        -StdErrPath (Join-Path $sampleDir "eval.stderr.log")

    if ($evalCode -ne 0 -or -not (Test-Path -LiteralPath $evalPath)) {
        $message = "Evaluation failed or output CSV missing."
        Write-WorkerStatus $statusPath $SampleIndex "failed" $evalCode $message
        exit 20
    }

    $reportArgs = @(
        "-m", "neuramem_benchmark.report",
        "--input", $evalPath,
        "--ingest-usage-dir", $sampleDir
    )
    $reportCode = Invoke-LoggedProcess `
        -FilePath $PythonPath `
        -ArgumentList $reportArgs `
        -StdOutPath (Join-Path $sampleDir "report.stdout.log") `
        -StdErrPath (Join-Path $sampleDir "report.stderr.log")

    if ($reportCode -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $sampleDir "summary.txt"))) {
        $message = "Report failed or summary missing."
        Write-WorkerStatus $statusPath $SampleIndex "failed" $reportCode $message
        exit 30
    }

    Write-WorkerStatus $statusPath $SampleIndex "completed" 0 "Ingest, evaluation, and report completed."
    exit 0
}

$BatchLogPath = Join-Path $OutputRoot "batch.log"
$StatePath = Join-Path $OutputRoot "state.json"
$LockPath = Join-Path $OutputRoot ".batch.lock"

function Write-BatchLog {
    param(
        [string]$Event,
        [string]$Message,
        [hashtable]$Fields = @{}
    )

    $parts = @(
        "ts=$([DateTime]::UtcNow.ToString('o'))",
        "module=worker.locomo-batch",
        "event=$Event",
        "message=`"$Message`""
    )
    foreach ($key in $Fields.Keys) {
        $parts += "$key=$($Fields[$key])"
    }
    Add-Content -LiteralPath $BatchLogPath -Value ($parts -join " ") -Encoding UTF8
}

function New-InitialState {
    return [ordered]@{
        total_samples = $SampleCount
        batch_size = $BatchSize
        completed = @()
        failed = @()
        running = @()
        last_batch = @()
        updated_utc = [DateTime]::UtcNow.ToString("o")
    }
}

function Read-State {
    if (-not (Test-Path -LiteralPath $StatePath)) {
        return New-InitialState
    }
    return Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Save-State {
    param([object]$State)

    $State.updated_utc = [DateTime]::UtcNow.ToString("o")
    $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Resolve-WorkerExitCode {
    param(
        [int]$Index,
        [System.Diagnostics.Process]$Process
    )

    $Process.Refresh()
    $rawExitCode = $null
    try {
        $rawExitCode = $Process.ExitCode
    } catch {
        $rawExitCode = $null
    }

    $statusPath = Join-Path $OutputRoot ("sample_{0:D2}\worker_status.json" -f $Index)
    if (Test-Path -LiteralPath $statusPath) {
        try {
            $workerStatus = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 |
                ConvertFrom-Json
            if ($workerStatus.status -eq "completed") {
                return 0
            }
            if ($workerStatus.status -eq "failed" -and $null -ne $workerStatus.exit_code) {
                return [int]$workerStatus.exit_code
            }
        } catch {
            Write-BatchLog "sample.status_read_failed" "Worker status could not be parsed; using process result." @{
                sample = $Index
            }
        }
    }

    if ($null -ne $rawExitCode) {
        return [int]$rawExitCode
    }
    return 1
}

try {
    $lockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
} catch {
    Write-BatchLog "batch.lock_skipped" "Another batch is already running."
    exit 0
}

try {
    $state = Read-State
    $completed = @($state.completed | ForEach-Object { [int]$_ })

    $requestedSampleIndices = @()
    if (-not [string]::IsNullOrWhiteSpace($SampleIndices)) {
        foreach ($rawIndex in ($SampleIndices -split ',')) {
            $parsedIndex = 0
            if ([int]::TryParse($rawIndex.Trim(), [ref]$parsedIndex)) {
                $requestedSampleIndices += $parsedIndex
            } else {
                Write-BatchLog "batch.selection_invalid" "Ignoring an invalid sample index." @{
                    value = $rawIndex.Trim()
                }
            }
        }
    }

    if ($requestedSampleIndices.Count -gt 0) {
        $selected = @($requestedSampleIndices | Sort-Object -Unique)
    } else {
        $selected = @(
            0..($SampleCount - 1) |
                Where-Object { $_ -notin $completed } |
                Select-Object -First $BatchSize
        )
    }

    $selected = @($selected | Where-Object { $_ -ge 0 -and $_ -lt $SampleCount })
    Write-BatchLog "batch.selection" "Resolved samples for this invocation." @{
        requested = ($requestedSampleIndices -join ',')
        completed = ($completed -join ',')
        sample_count = $SampleCount
        batch_size = $BatchSize
        selected = ($selected -join ',')
    }
    if ($selected.Count -eq 0) {
        Write-BatchLog "batch.all_completed" "All samples are already complete." @{ total_samples = $SampleCount }
        exit 0
    }

    $state.running = $selected
    $state.last_batch = $selected
    Save-State $state
    Write-BatchLog "batch.started" "Starting parallel sample batch." @{
        batch = ($selected -join ',')
        total_samples = $SampleCount
    }

    $shellPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    $workers = @()
    foreach ($index in $selected) {
        $sampleDir = Join-Path $OutputRoot ("sample_{0:D2}" -f $index)
        New-Item -ItemType Directory -Path $sampleDir -Force | Out-Null
        $workerArguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$PSCommandPath`"",
            "-Worker",
            "-SampleIndex", $index,
            "-SampleCount", $SampleCount,
            "-DataPath", "`"$DataPath`"",
            "-OutputRoot", "`"$OutputRoot`"",
            "-PythonPath", "`"$PythonPath`"",
            "-Threads", $Threads
        )
        $process = Start-Process `
            -FilePath $shellPath `
            -ArgumentList $workerArguments `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput (Join-Path $sampleDir "worker.stdout.log") `
            -RedirectStandardError (Join-Path $sampleDir "worker.stderr.log") `
            -PassThru
        $workers += [pscustomobject]@{ Index = $index; Process = $process }
        Write-BatchLog "sample.started" "Sample worker started." @{
            sample = $index
            pid = $process.Id
        }
    }

    $results = @()
    foreach ($workerEntry in $workers) {
        $workerEntry.Process.WaitForExit()
        $exitCode = Resolve-WorkerExitCode $workerEntry.Index $workerEntry.Process
        $results += [pscustomobject]@{ Index = $workerEntry.Index; ExitCode = $exitCode }
        if ($exitCode -eq 0) {
            Write-BatchLog "sample.completed" "Sample worker completed." @{
                sample = $workerEntry.Index
                exit_code = $exitCode
            }
        } else {
            Write-BatchLog "sample.failed" "Sample worker failed." @{
                sample = $workerEntry.Index
                exit_code = $exitCode
            }
        }
    }

    $completed = @($completed + @($results | Where-Object { $_.ExitCode -eq 0 } | Select-Object -ExpandProperty Index) | Sort-Object -Unique)
    $failed = @($state.failed | ForEach-Object { [int]$_ })
    $failed = @($failed + @($results | Where-Object { $_.ExitCode -ne 0 } | Select-Object -ExpandProperty Index) | Sort-Object -Unique)
    $failed = @($failed | Where-Object { $_ -notin $completed })
    $state.completed = $completed
    $state.failed = $failed
    $state.running = @()
    Save-State $state

    $failedText = if ($failed.Count -gt 0) { $failed -join ',' } else { "none" }
    Write-BatchLog "batch.completed" "Parallel sample batch completed." @{
        completed = ($completed -join ',')
        failed = $failedText
    }
    if (@($results | Where-Object { $_.ExitCode -ne 0 }).Count -gt 0) {
        exit 1
    }
} finally {
    if ($null -ne $lockStream) {
        $lockStream.Dispose()
    }
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}
