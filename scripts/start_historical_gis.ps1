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
    if (Assert-PortIdentity 8000 ${function:Test-Backend} "Historical GIS backend") {
        Write-Host "      Already running on 8000."
    }
    else {
        Write-Host "      Starting Historical GIS backend..."
        Start-HiddenService "backend" $pythonExe @(
            "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000",
            "--env-file", (Quote-ProcessArgument $externalEnv), "--app-dir", (Quote-ProcessArgument $projectRoot)
        ) $projectRoot | Out-Null
        Wait-Service ${function:Test-Backend} "Backend"
        Write-Host "      Backend ready."
    }

    Write-Host "[3/4] Checking frontend..."
    if (Assert-PortIdentity 5173 ${function:Test-Frontend} "Historical GIS frontend") {
        Write-Host "      Already running on 5173."
    }
    else {
        Write-Host "      Starting frontend..."
        Start-HiddenService "frontend" $pnpm.Source @(
            "run", "dev", "--port", "5173", "--strictPort"
        ) $frontendRoot | Out-Null
        Wait-Service ${function:Test-Frontend} "Frontend"
        Write-Host "      Frontend ready."
    }

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
