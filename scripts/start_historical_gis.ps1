[CmdletBinding()]
param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$sharedRuntime = "C:\D\python\202608231533"
$pythonExe = Join-Path $sharedRuntime ".venv\Scripts\python.exe"
$chromaExe = Join-Path $sharedRuntime ".venv\Scripts\chroma.exe"
$externalEnv = Join-Path $sharedRuntime ".env"
$chromaData = Join-Path $sharedRuntime "data\chroma_server_roman_republic_v2"
$frontendRoot = Join-Path $projectRoot "frontend"
$frontendModules = Join-Path $frontendRoot "node_modules"
$runtimeRoot = Join-Path $env:LOCALAPPDATA "HistoricalGISAgent\runtime"
$collectionName = "roman_republic_primary_sources_v2"
$browserUrl = "http://127.0.0.1:5173"
$startedProcesses = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

function Assert-Path {
    param([string]$Path, [string]$Description, [ValidateSet("Leaf", "Container")][string]$PathType)

    if (-not (Test-Path -LiteralPath $Path -PathType $PathType)) {
        throw "$Description is missing: $Path"
    }
}

function Normalize-ProjectPath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\').ToLowerInvariant()
}

function Get-PortListenerProcess {
    param([int]$Port)

    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $connection) {
        return $null
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)" -ErrorAction SilentlyContinue
    if (-not $process) {
        return $null
    }
    return [pscustomobject]@{
        ProcessId = [int]$process.ProcessId
        CommandLine = [string]$process.CommandLine
    }
}

function Test-IsHistoricalGisBackendCommand {
    param([string]$CommandLine)

    return ($CommandLine -match "uvicorn" -and $CommandLine -match "backend\.app\.main:app")
}

function Get-BackendAppDirFromCommandLine {
    param([string]$CommandLine)

    if ($CommandLine -match '--app-dir\s+"([^"]+)"') {
        return $Matches[1]
    }
    if ($CommandLine -match "--app-dir\s+([^\s]+)") {
        return $Matches[1]
    }
    return $null
}

function Test-IsHistoricalGisFrontendCommand {
    param([string]$CommandLine)

    return ($CommandLine -match "vite(\.js)?" -and $CommandLine -match "5173")
}

function Get-FrontendRootFromCommandLine {
    param([string]$CommandLine)

    if ($CommandLine -match '(?<frontend>[A-Za-z]:\\.+\\historical-gis-[^\\]+\\frontend)') {
        return $Matches["frontend"]
    }
    return $null
}

function Resolve-HistoricalGisBackendPort {
    param(
        [int]$Port,
        [string]$ExpectedProjectRoot
    )

    $listener = Get-PortListenerProcess $Port
    if (-not $listener) {
        return [pscustomobject]@{ Status = "Free" }
    }
    if (-not (Test-IsHistoricalGisBackendCommand $listener.CommandLine)) {
        return [pscustomobject]@{
            Status = "UnknownOccupant"
            ProcessId = $listener.ProcessId
            CommandLine = $listener.CommandLine
        }
    }

    $appDir = Get-BackendAppDirFromCommandLine $listener.CommandLine
    if (-not $appDir) {
        return [pscustomobject]@{
            Status = "UnverifiedHistoricalGis"
            ProcessId = $listener.ProcessId
            CommandLine = $listener.CommandLine
        }
    }

    $expected = Normalize-ProjectPath $ExpectedProjectRoot
    $actual = Normalize-ProjectPath $appDir
    if ($actual -eq $expected) {
        return [pscustomobject]@{
            Status = "SameWorktree"
            ProcessId = $listener.ProcessId
            AppDir = $appDir
        }
    }

    return [pscustomobject]@{
        Status = "StaleWorktree"
        ProcessId = $listener.ProcessId
        ExpectedProjectRoot = $ExpectedProjectRoot
        ActualProjectRoot = $appDir
        CommandLine = $listener.CommandLine
    }
}

function Resolve-HistoricalGisFrontendPort {
    param(
        [int]$Port,
        [string]$ExpectedFrontendRoot
    )

    $listener = Get-PortListenerProcess $Port
    if (-not $listener) {
        return [pscustomobject]@{ Status = "Free" }
    }
    if (-not (Test-IsHistoricalGisFrontendCommand $listener.CommandLine)) {
        return [pscustomobject]@{
            Status = "UnknownOccupant"
            ProcessId = $listener.ProcessId
            CommandLine = $listener.CommandLine
        }
    }

    $frontendRootFromCommand = Get-FrontendRootFromCommandLine $listener.CommandLine
    if (-not $frontendRootFromCommand) {
        return [pscustomobject]@{
            Status = "UnverifiedHistoricalGis"
            ProcessId = $listener.ProcessId
            CommandLine = $listener.CommandLine
        }
    }

    $expected = Normalize-ProjectPath $ExpectedFrontendRoot
    $actual = Normalize-ProjectPath $frontendRootFromCommand
    if ($actual -eq $expected) {
        return [pscustomobject]@{
            Status = "SameWorktree"
            ProcessId = $listener.ProcessId
            FrontendRoot = $frontendRootFromCommand
        }
    }

    return [pscustomobject]@{
        Status = "StaleWorktree"
        ProcessId = $listener.ProcessId
        ExpectedFrontendRoot = $ExpectedFrontendRoot
        ActualFrontendRoot = $frontendRootFromCommand
        CommandLine = $listener.CommandLine
    }
}

function Stop-StaleHistoricalGisProcess {
    param(
        [string]$ServiceName,
        [int]$Port,
        [pscustomobject]$Occupancy
    )

    $expected = if ($Occupancy.ExpectedProjectRoot) { $Occupancy.ExpectedProjectRoot } else { $Occupancy.ExpectedFrontendRoot }
    $actual = if ($Occupancy.ActualProjectRoot) { $Occupancy.ActualProjectRoot } else { $Occupancy.ActualFrontendRoot }
    Write-Host "      Stale Historical GIS runtime detected ($ServiceName)"
    Write-Host "      Expected: $expected"
    Write-Host "      Actual:   $actual"
    Stop-Process -Id $Occupancy.ProcessId -ErrorAction Stop
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        if (-not (Test-PortListening $Port)) {
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "Stale $ServiceName process did not release port $Port in time."
}

function Test-PortListening {
    param([int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        return $client.ConnectAsync("127.0.0.1", $Port).Wait(500)
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Invoke-LocalJson {
    param([string]$Uri)

    try {
        return Invoke-RestMethod -Uri $Uri -TimeoutSec 3 -ErrorAction Stop
    }
    catch {
        return $null
    }
}

function Test-Chroma {
    $response = Invoke-LocalJson "http://127.0.0.1:8002/api/v2/heartbeat"
    return $null -ne $response -and
        $null -ne $response.PSObject.Properties["nanosecond heartbeat"]
}

function Test-ChromaCollection {
    $uri = "http://127.0.0.1:8002/api/v2/tenants/default_tenant/databases/default_database/collections"
    $collections = Invoke-LocalJson $uri
    return $null -ne $collections -and $collectionName -in @($collections.name)
}

function Get-ChromaCollectionCount {
    $uri = "http://127.0.0.1:8002/api/v2/tenants/default_tenant/databases/default_database/collections"
    $collections = Invoke-LocalJson $uri
    $collection = $collections | Where-Object { $_.name -eq $collectionName } | Select-Object -First 1
    if (-not $collection) {
        return $null
    }
    $countUri = "http://127.0.0.1:8002/api/v2/tenants/default_tenant/databases/default_database/collections/$($collection.id)/count"
    $countResponse = Invoke-LocalJson $countUri
    return $countResponse.count
}

function Test-Backend {
    $response = Invoke-LocalJson "http://127.0.0.1:8000/health"
    return $null -ne $response -and $response.status -eq "ok" -and
        $null -ne $response.agent -and $null -ne $response.provider
}

function Test-Frontend {
    try {
        $response = Invoke-WebRequest -Uri $browserUrl -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        return $response.StatusCode -eq 200 -and $response.Content -match 'id=["'']root["'']' -and
            $response.Content -match '/src/main\.tsx'
    }
    catch {
        return $false
    }
}

function Wait-Service {
    param(
        [scriptblock]$Probe,
        [string]$Name,
        [int]$Attempts = 60
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        if (& $Probe) {
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "$Name did not become ready after $Attempts seconds. Check the logs in $runtimeRoot."
}

function Start-HiddenService {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )

    $stdout = Join-Path $runtimeRoot "$Name.stdout.log"
    $stderr = Join-Path $runtimeRoot "$Name.stderr.log"
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $startedProcesses.Add($process)
    return $process
}

function Quote-ProcessArgument {
    param([string]$Value)

    return '"' + $Value.Replace('"', '\"') + '"'
}

function Stop-ProcessesStartedThisRun {
    foreach ($process in $startedProcesses) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        }
    }
}

function Ensure-HistoricalGisBackend {
    param([string]$ExpectedProjectRoot)

    $occupancy = Resolve-HistoricalGisBackendPort 8000 $ExpectedProjectRoot
    switch ($occupancy.Status) {
        "Free" {
            Write-Host "      Starting Historical GIS backend..."
            Start-HiddenService "backend" $pythonExe @(
                "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000",
                "--env-file", (Quote-ProcessArgument $externalEnv), "--app-dir", (Quote-ProcessArgument $ExpectedProjectRoot)
            ) $ExpectedProjectRoot | Out-Null
            Wait-Service ${function:Test-Backend} "Backend"
            Write-Host "      Backend ready."
        }
        "SameWorktree" {
            if (-not (Test-Backend)) {
                throw "Port 8000 belongs to the expected Historical GIS backend worktree, but the health probe failed."
            }
            Write-Host "      Already running on 8000 (same worktree)."
        }
        "StaleWorktree" {
            Stop-StaleHistoricalGisProcess "backend" 8000 $occupancy
            Ensure-HistoricalGisBackend $ExpectedProjectRoot
        }
        "UnknownOccupant" {
            throw "Port 8000 is occupied by an unrelated process (PID $($occupancy.ProcessId))."
        }
        "UnverifiedHistoricalGis" {
            throw "Port 8000 has a Historical GIS backend, but its worktree could not be verified from --app-dir."
        }
        default {
            throw "Unexpected backend port occupancy state: $($occupancy.Status)"
        }
    }
}

function Ensure-HistoricalGisFrontend {
    param([string]$ExpectedFrontendRoot)

    $occupancy = Resolve-HistoricalGisFrontendPort 5173 $ExpectedFrontendRoot
    switch ($occupancy.Status) {
        "Free" {
            Write-Host "      Starting frontend..."
            Start-HiddenService "frontend" $pnpm.Source @(
                "run", "dev", "--port", "5173", "--strictPort"
            ) $ExpectedFrontendRoot | Out-Null
            Wait-Service ${function:Test-Frontend} "Frontend"
            Write-Host "      Frontend ready."
        }
        "SameWorktree" {
            if (-not (Test-Frontend)) {
                throw "Port 5173 belongs to the expected Historical GIS frontend worktree, but the health probe failed."
            }
            Write-Host "      Already running on 5173 (same worktree)."
        }
        "StaleWorktree" {
            Stop-StaleHistoricalGisProcess "frontend" 5173 $occupancy
            Ensure-HistoricalGisFrontend $ExpectedFrontendRoot
        }
        "UnknownOccupant" {
            throw "Port 5173 is occupied by an unrelated process (PID $($occupancy.ProcessId))."
        }
        "UnverifiedHistoricalGis" {
            throw "Port 5173 has a Historical GIS frontend, but existing frontend identity could not be verified."
        }
        default {
            throw "Unexpected frontend port occupancy state: $($occupancy.Status)"
        }
    }
}

function Assert-PortIdentity {
    param(
        [int]$Port,
        [scriptblock]$Probe,
        [string]$Name
    )

    if (-not (Test-PortListening $Port)) {
        return $false
    }
    if (-not (& $Probe)) {
        throw "Port $Port is already occupied and does not appear to be the expected $Name service."
    }
    return $true
}

if ($MyInvocation.InvocationName -eq '.') {
    return
}

try {
    Assert-Path $pythonExe "Shared Python interpreter" Leaf
    Assert-Path $chromaExe "Chroma executable" Leaf
    Assert-Path $externalEnv "Canonical external .env" Leaf
    Assert-Path $chromaData "Canonical Chroma persistence" Container
    Assert-Path (Join-Path $frontendRoot "package.json") "Frontend package.json" Leaf
    Assert-Path $frontendModules "Frontend dependencies (run 'pnpm install --frozen-lockfile' in frontend)" Container

    $pnpm = Get-Command pnpm.cmd -ErrorAction SilentlyContinue
    if ($null -eq $pnpm) {
        throw "pnpm is unavailable. Install pnpm, then retry."
    }
    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

    Write-Host "[1/4] Checking Chroma..."
    if (Assert-PortIdentity 8002 ${function:Test-Chroma} "Chroma") {
        Write-Host "      Already running on 8002."
    }
    else {
        Write-Host "      Starting the existing Roman Republic Chroma store..."
        Start-HiddenService "chroma" $chromaExe @(
            "run", "--path", (Quote-ProcessArgument $chromaData), "--host", "127.0.0.1", "--port", "8002"
        ) $projectRoot | Out-Null
        Wait-Service ${function:Test-Chroma} "Chroma"
        Write-Host "      Chroma ready."
    }
    if (-not (Test-ChromaCollection)) {
        throw "Chroma is running, but expected collection '$collectionName' is unavailable."
    }

    $env:RAG_CHROMA_HOST = "127.0.0.1"
    $env:RAG_CHROMA_PORT = "8002"
    $env:RAG_COLLECTION = $collectionName
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONPATH = $projectRoot

    Write-Host "[2/4] Checking backend..."
    Ensure-HistoricalGisBackend $projectRoot

    Write-Host "[3/4] Checking frontend..."
    Ensure-HistoricalGisFrontend $frontendRoot

    Write-Host "[4/4] Opening browser..."
    if (-not $NoBrowser) {
        Start-Process $browserUrl
    }
    else {
        Write-Host "      Browser opening skipped by -NoBrowser."
    }
    Write-Host ""
    Write-Host "Historical GIS Agent is ready:"
    Write-Host $browserUrl
    exit 0
}
catch {
    Write-Host ""
    Write-Error $_.Exception.Message
    Stop-ProcessesStartedThisRun
    exit 1
}
