[CmdletBinding()]
param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$DefaultSharedRuntime = "C:\D\python\202608231533"
$sharedRuntime = $DefaultSharedRuntime
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
    return $countResponse
}

function Test-Backend {
    $response = Invoke-LocalJson "http://127.0.0.1:8000/health"
    return $null -ne $response -and $response.status -eq "ok" -and
        $null -ne $response.agent -and $null -ne $response.provider
}

function Assert-BackendGisAssets {
    $response = Invoke-LocalJson "http://127.0.0.1:8000/health"
    $required = @("pleiades", "srtm", "natural_earth")
    if (Test-EnvFlagEnabled "ROMAN_ROAD_ENABLED") {
        $required += "itiner_e"
    }
    foreach ($asset in $required) {
        if ($null -eq $response -or $response.$asset -ne "ACTIVE") {
            throw "Backend GIS asset '$asset' is not ACTIVE. A backend already running on 8000 may predate this configuration; restart that matching backend and relaunch."
        }
    }
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

function Get-LauncherEnvValue {
    param([string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $null
    }
    return $value.Trim()
}

function Convert-ToFullPath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-EnvFlagEnabled {
    param([string]$Name)

    $value = Get-LauncherEnvValue $Name
    if ($null -eq $value) {
        return $false
    }
    return $value -match '^(1|true|yes|on)$'
}

function Resolve-RequiredConfiguredPath {
    param(
        [string]$EnvName,
        [string]$DerivedPath,
        [string]$Description,
        [ValidateSet("Leaf", "Container")][string]$PathType
    )

    $explicit = Get-LauncherEnvValue $EnvName
    if ($null -ne $explicit) {
        $full = Convert-ToFullPath $explicit
        if (-not (Test-Path -LiteralPath $full -PathType $PathType)) {
            throw "$Description is missing (explicit $EnvName). The original-machine fallback was not used: $full"
        }
        return $full
    }
    if (-not (Test-Path -LiteralPath $DerivedPath -PathType $PathType)) {
        throw "$Description is missing: $DerivedPath. Set $EnvName or HISTORICAL_GIS_RUNTIME_ROOT to a complete runtime."
    }
    return (Convert-ToFullPath $DerivedPath)
}

function Resolve-HistoricalGisLauncherConfig {
    param([switch]$SkipEmbeddingCache)

    $explicitRuntime = Get-LauncherEnvValue "HISTORICAL_GIS_RUNTIME_ROOT"
    if ($null -ne $explicitRuntime) {
        $script:sharedRuntime = Convert-ToFullPath $explicitRuntime
        if (-not (Test-Path -LiteralPath $script:sharedRuntime -PathType Container)) {
            throw "HISTORICAL_GIS_RUNTIME_ROOT is missing: $($script:sharedRuntime)"
        }
        $script:usedLegacyRuntimeFallback = $false
    }
    elseif (Test-Path -LiteralPath $DefaultSharedRuntime -PathType Container) {
        $script:sharedRuntime = Convert-ToFullPath $DefaultSharedRuntime
        $script:usedLegacyRuntimeFallback = $true
    }
    else {
        throw "HISTORICAL_GIS_RUNTIME_ROOT is unset and the original-machine runtime is absent: $DefaultSharedRuntime"
    }

    $script:pythonExe = Resolve-RequiredConfiguredPath "HISTORICAL_GIS_PYTHON" (Join-Path $script:sharedRuntime ".venv\Scripts\python.exe") "Python executable" Leaf
    $script:chromaExe = Resolve-RequiredConfiguredPath "HISTORICAL_GIS_CHROMA_EXE" (Join-Path $script:sharedRuntime ".venv\Scripts\chroma.exe") "Chroma executable" Leaf
    $script:externalEnv = Resolve-RequiredConfiguredPath "HISTORICAL_GIS_ENV_FILE" (Join-Path $script:sharedRuntime ".env") "Environment file" Leaf
    $script:chromaData = Resolve-RequiredConfiguredPath "HISTORICAL_GIS_CHROMA_DATA" (Join-Path $script:sharedRuntime "data\chroma_server_roman_republic_v2") "Chroma persistence directory" Container

    $logRoot = Get-LauncherEnvValue "HISTORICAL_GIS_LOG_ROOT"
    if ($null -ne $logRoot) {
        $script:runtimeRoot = Convert-ToFullPath $logRoot
    }

    $pleiades = Get-LauncherEnvValue "PLEIADES_GAZETTEER_PATH"
    if ($null -ne $pleiades) {
        $pleiadesFull = Convert-ToFullPath $pleiades
        if (-not (Test-Path -LiteralPath $pleiadesFull -PathType Leaf)) {
            throw "PLEIADES_GAZETTEER_PATH is missing: $pleiadesFull"
        }
        $env:PLEIADES_GAZETTEER_PATH = $pleiadesFull
    }

    $script:forceOfflineModels = $false
    if ($null -eq (Get-LauncherEnvValue "HF_HUB_OFFLINE") -and $script:usedLegacyRuntimeFallback) {
        $script:forceOfflineModels = $true
    }

    if (-not $SkipEmbeddingCache -and $script:forceOfflineModels) {
        $cacheHint = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--intfloat--multilingual-e5-small"
        if (-not (Test-Path -LiteralPath $cacheHint -PathType Container)) {
            throw "Offline embedding cache is unavailable at $cacheHint. Place intfloat/multilingual-e5-small in the local Hugging Face hub cache, or set HF_HUB_OFFLINE before launch. The launcher will not download weights."
        }
    }
}

function Assert-OptionalGisDependencies {
    if (Test-EnvFlagEnabled "ROMAN_ROAD_ENABLED") {
        $geojson = Get-LauncherEnvValue "ROMAN_ROAD_GEOJSON_PATH"
        if ($null -eq $geojson) {
            throw "ROMAN_ROAD_ENABLED is true, but ROMAN_ROAD_GEOJSON_PATH is not set."
        }
        $geojsonFull = Convert-ToFullPath $geojson
        if (-not (Test-Path -LiteralPath $geojsonFull -PathType Leaf)) {
            throw "ROMAN_ROAD_GEOJSON_PATH is missing: $geojsonFull"
        }
        $env:ROMAN_ROAD_GEOJSON_PATH = $geojsonFull
    }

    $maritime = Get-LauncherEnvValue "MARITIME_SURFACE_DATA_ROOT"
    if ($null -ne $maritime) {
        $maritimeFull = Convert-ToFullPath $maritime
        if (-not (Test-Path -LiteralPath $maritimeFull -PathType Container)) {
            throw "MARITIME_SURFACE_DATA_ROOT is missing: $maritimeFull"
        }
        $env:MARITIME_SURFACE_DATA_ROOT = $maritimeFull
    }
}

function Enable-DefaultGisAssets {
    # Explicit deployment overrides, including ROMAN_ROAD_ENABLED=0, take precedence.
    if ($null -eq (Get-LauncherEnvValue "PLEIADES_GAZETTEER_PATH")) {
        $env:PLEIADES_GAZETTEER_PATH = Join-Path $projectRoot "data\pleiades_v4_1\pleiades_v4_1.sqlite3"
    }
    if ($null -eq (Get-LauncherEnvValue "ROMAN_ROAD_ENABLED")) {
        $env:ROMAN_ROAD_ENABLED = "1"
    }
    if ((Test-EnvFlagEnabled "ROMAN_ROAD_ENABLED") -and $null -eq (Get-LauncherEnvValue "ROMAN_ROAD_GEOJSON_PATH")) {
        $env:ROMAN_ROAD_GEOJSON_PATH = Join-Path $projectRoot "data\raw\itiner_e\itinere_roads_zenodo_17122148.geojson"
    }
    if ($null -eq (Get-LauncherEnvValue "MARITIME_SURFACE_DATA_ROOT")) {
        $env:MARITIME_SURFACE_DATA_ROOT = Join-Path $projectRoot "data\gis\natural_earth_10m"
    }
    Assert-Path $env:PLEIADES_GAZETTEER_PATH "Pleiades index" Leaf
    Assert-Path (Join-Path $env:MARITIME_SURFACE_DATA_ROOT "surface-manifest.json") "Natural Earth manifest" Leaf
}

function Get-ChromaPathFromCommandLine {
    param([string]$CommandLine)

    if ($CommandLine -match '--path\s+"([^"]+)"') {
        return $Matches[1]
    }
    if ($CommandLine -match "--path\s+([^\s]+)") {
        return $Matches[1]
    }
    return $null
}

function Resolve-HistoricalGisChromaPort {
    param(
        [int]$Port,
        [string]$ExpectedChromaData
    )

    $listener = Get-PortListenerProcess $Port
    if (-not $listener) {
        return [pscustomobject]@{ Status = "Free" }
    }
    if ($listener.CommandLine -notmatch "chroma") {
        return [pscustomobject]@{
            Status = "UnknownOccupant"
            ProcessId = $listener.ProcessId
            CommandLine = $listener.CommandLine
        }
    }
    $actualPath = Get-ChromaPathFromCommandLine $listener.CommandLine
    if (-not $actualPath) {
        return [pscustomobject]@{
            Status = "UnverifiedChroma"
            ProcessId = $listener.ProcessId
            CommandLine = $listener.CommandLine
        }
    }
    if ((Normalize-ProjectPath $actualPath) -eq (Normalize-ProjectPath $ExpectedChromaData)) {
        return [pscustomobject]@{
            Status = "SameConfig"
            ProcessId = $listener.ProcessId
            ChromaData = $actualPath
        }
    }
    return [pscustomobject]@{
        Status = "IncompatibleRuntime"
        ProcessId = $listener.ProcessId
        ExpectedChromaData = $ExpectedChromaData
        ActualChromaData = $actualPath
        CommandLine = $listener.CommandLine
    }
}

function Ensure-HistoricalGisChroma {
    param([string]$ExpectedChromaData)

    $occupancy = Resolve-HistoricalGisChromaPort 8002 $ExpectedChromaData
    switch ($occupancy.Status) {
        "Free" {
            Write-Host "      Starting the configured Roman Republic Chroma store..."
            Start-HiddenService "chroma" $chromaExe @(
                "run", "--path", (Quote-ProcessArgument $ExpectedChromaData), "--host", "127.0.0.1", "--port", "8002"
            ) $projectRoot | Out-Null
            Wait-Service ${function:Test-Chroma} "Chroma"
            Write-Host "      Chroma ready."
        }
        "SameConfig" {
            if (-not (Test-Chroma)) {
                throw "Port 8002 belongs to Chroma with the expected persistence path, but the heartbeat probe failed."
            }
            Write-Host "      Already running on 8002 (same Chroma persistence)."
        }
        "IncompatibleRuntime" {
            throw "Port 8002 is Chroma using a different persistence path ($($occupancy.ActualChromaData)). It was not stopped. Use a free port or matching HISTORICAL_GIS_CHROMA_DATA."
        }
        "UnknownOccupant" {
            throw "Port 8002 is occupied by an unrelated process (PID $($occupancy.ProcessId))."
        }
        "UnverifiedChroma" {
            throw "Port 8002 has a Chroma process, but --path could not be verified from the command line."
        }
        default {
            throw "Unexpected Chroma port occupancy state: $($occupancy.Status)"
        }
    }
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
            Assert-BackendGisAssets
            Write-Host "      Backend ready."
        }
        "SameWorktree" {
            if (-not (Test-Backend)) {
                throw "Port 8000 belongs to the expected Historical GIS backend worktree, but the health probe failed."
            }
            Assert-BackendGisAssets
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
    Resolve-HistoricalGisLauncherConfig
    Enable-DefaultGisAssets
    Assert-OptionalGisDependencies
    Assert-Path (Join-Path $frontendRoot "package.json") "Frontend package.json" Leaf
    Assert-Path $frontendModules "Frontend dependencies (run 'pnpm install --frozen-lockfile' in frontend)" Container

    $pnpm = Get-Command pnpm.cmd -ErrorAction SilentlyContinue
    if ($null -eq $pnpm) {
        throw "pnpm is unavailable. Install pnpm, then retry."
    }
    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

    Write-Host "[1/4] Checking Chroma..."
    Ensure-HistoricalGisChroma $chromaData
    if (-not (Test-ChromaCollection)) {
        throw "Chroma is running, but expected collection '$collectionName' is unavailable."
    }

    $env:RAG_CHROMA_HOST = "127.0.0.1"
    $env:RAG_CHROMA_PORT = "8002"
    $env:RAG_COLLECTION = $collectionName
    if ($script:forceOfflineModels) {
        $env:HF_HUB_OFFLINE = "1"
        $env:TRANSFORMERS_OFFLINE = "1"
    }
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
