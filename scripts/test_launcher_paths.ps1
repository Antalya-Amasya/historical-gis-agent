$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "start_historical_gis.ps1")

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Assert-Contains {
    param([string]$Haystack, [string]$Needle, [string]$Message)
    if ($Haystack -notlike "*$Needle*") {
        throw "$Message (missing '$Needle' in '$Haystack')"
    }
}

function Assert-NotContains {
    param([string]$Haystack, [string]$Needle, [string]$Message)
    if ($Haystack -like "*$Needle*") {
        throw "$Message (unexpected '$Needle' in '$Haystack')"
    }
}

function Assert-SamePath {
    param([string]$Expected, [string]$Actual, [string]$Message)
    if ((Normalize-ProjectPath $Expected) -ne (Normalize-ProjectPath $Actual)) {
        throw "$Message (expected '$Expected', got '$Actual')"
    }
}

function Assert-PathUnder {
    param([string]$Root, [string]$Path, [string]$Message)
    $rootN = Normalize-ProjectPath $Root
    $pathN = Normalize-ProjectPath $Path
    if (-not $pathN.StartsWith($rootN)) {
        throw "$Message (path '$Path' is not under '$Root')"
    }
}

$saved = @{}
@(
    "HISTORICAL_GIS_RUNTIME_ROOT",
    "HISTORICAL_GIS_PYTHON",
    "HISTORICAL_GIS_CHROMA_EXE",
    "HISTORICAL_GIS_ENV_FILE",
    "HISTORICAL_GIS_CHROMA_DATA",
    "HISTORICAL_GIS_LOG_ROOT",
    "PLEIADES_GAZETTEER_PATH",
    "ROMAN_ROAD_ENABLED",
    "ROMAN_ROAD_GEOJSON_PATH",
    "MARITIME_SURFACE_DATA_ROOT",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE"
) | ForEach-Object {
    $saved[$_] = [Environment]::GetEnvironmentVariable($_)
    [Environment]::SetEnvironmentVariable($_, $null)
}

$tempRoot = Join-Path $env:TEMP ("historical-gis-rc11-" + [guid]::NewGuid().ToString("N"))
$standardRoot = Join-Path $tempRoot "standard"
$legacyRoot = Join-Path $tempRoot "legacy"
$explicitRoot = Join-Path $tempRoot "explicit"
$script:DefaultSharedRuntime = $standardRoot
$script:LegacySharedRuntime = $legacyRoot

try {
    foreach ($runtimeRoot in @($standardRoot, $legacyRoot, $explicitRoot)) {
        New-Item -ItemType Directory -Path (Join-Path $runtimeRoot ".venv\Scripts") -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $runtimeRoot "data\chroma_server_roman_republic_v2") -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $runtimeRoot ".venv\Scripts\python.exe") -Value "" -Encoding ascii
        Set-Content -LiteralPath (Join-Path $runtimeRoot ".venv\Scripts\chroma.exe") -Value "" -Encoding ascii
        Set-Content -LiteralPath (Join-Path $runtimeRoot ".env") -Value "LLM_PROVIDER=fake" -Encoding ascii
    }

    # A. Prefer the standard runtime root when it exists.
    Resolve-HistoricalGisLauncherConfig -SkipEmbeddingCache
    Assert-True (-not $script:usedLegacyRuntimeFallback) "standard runtime must not be marked as a legacy fallback"
    Assert-True $script:usedDefaultRuntime "standard runtime should be selected as the default"
    Assert-PathUnder $standardRoot $script:pythonExe "standard python should come from the standard runtime"
    Assert-PathUnder $standardRoot $script:chromaData "standard chroma data path"
    Assert-True (-not (Test-EnvFlagEnabled "ROMAN_ROAD_ENABLED")) "roman roads stay disabled unless explicitly enabled"
    Assert-OptionalGisDependencies
    Enable-DefaultGisAssets
    Assert-OptionalGisDependencies
    Assert-True (Test-EnvFlagEnabled "ROMAN_ROAD_ENABLED") "normal launcher enables existing Roman roads"
    Assert-SamePath (Join-Path $projectRoot "data\raw\itiner_e\itinere_roads_zenodo_17122148.geojson") $env:ROMAN_ROAD_GEOJSON_PATH "default road dataset"
    Assert-SamePath (Join-Path $projectRoot "data\gis\natural_earth_10m") $env:MARITIME_SURFACE_DATA_ROOT "default Natural Earth dataset"
    Assert-SamePath (Join-Path $projectRoot "data\pleiades_v4_1\pleiades_v4_1.sqlite3") $env:PLEIADES_GAZETTEER_PATH "default Pleiades index"
    @("PLEIADES_GAZETTEER_PATH", "ROMAN_ROAD_ENABLED", "ROMAN_ROAD_GEOJSON_PATH", "MARITIME_SURFACE_DATA_ROOT") | ForEach-Object {
        [Environment]::SetEnvironmentVariable($_, $null)
    }

    # B. Preserve the legacy fallback when the standard root is absent.
    $script:DefaultSharedRuntime = Join-Path $tempRoot "missing-standard"
    Resolve-HistoricalGisLauncherConfig -SkipEmbeddingCache
    Assert-True $script:usedLegacyRuntimeFallback "legacy runtime should be marked when the standard root is absent"
    Assert-True $script:usedDefaultRuntime "legacy fallback is still a default-selected runtime"
    Assert-PathUnder $legacyRoot $script:pythonExe "fallback python should come from the legacy runtime"
    Assert-PathUnder $legacyRoot $script:chromaData "fallback Chroma data should come from the legacy runtime"
    $script:DefaultSharedRuntime = $standardRoot

    # C. A runtime root supplied entirely through configuration.
    $env:HISTORICAL_GIS_RUNTIME_ROOT = $explicitRoot
    Resolve-HistoricalGisLauncherConfig -SkipEmbeddingCache
    Assert-True (-not $script:usedLegacyRuntimeFallback) "explicit runtime root must not use the original-machine fallback flag"
    Assert-PathUnder $explicitRoot $script:pythonExe "configured python must come from the configured runtime"
    Assert-PathUnder $explicitRoot $script:chromaExe "configured chroma exe must come from the configured runtime"
    Assert-PathUnder $explicitRoot $script:externalEnv "configured env file must come from the configured runtime"
    Assert-PathUnder $explicitRoot $script:chromaData "configured chroma data must come from the configured runtime"
    Assert-True ((Normalize-ProjectPath $script:pythonExe) -notlike ("*" + (Normalize-ProjectPath $legacyRoot) + "*")) "configured python must not read the legacy venv"
    Assert-True ((Normalize-ProjectPath $script:chromaData) -notlike ("*" + (Normalize-ProjectPath $legacyRoot) + "*")) "configured Chroma data must not read the legacy corpus"

    # Missing explicit Python must not fall back to the original machine.
    $env:HISTORICAL_GIS_PYTHON = Join-Path $explicitRoot "missing-python.exe"
    $pythonFailed = $false
    try {
        Resolve-HistoricalGisLauncherConfig -SkipEmbeddingCache
    }
    catch {
        $pythonFailed = $true
        Assert-Contains $_.Exception.Message "HISTORICAL_GIS_PYTHON" "missing python error names the override"
        Assert-Contains $_.Exception.Message "original-machine fallback was not used" "missing explicit python must not silently use the original runtime"
    }
    Assert-True $pythonFailed "missing explicit python must throw"
    [Environment]::SetEnvironmentVariable("HISTORICAL_GIS_PYTHON", $null)

    $env:HISTORICAL_GIS_CHROMA_EXE = Join-Path $explicitRoot "missing-chroma.exe"
    $chromaExeFailed = $false
    try {
        Resolve-HistoricalGisLauncherConfig -SkipEmbeddingCache
    }
    catch {
        $chromaExeFailed = $true
        Assert-Contains $_.Exception.Message "HISTORICAL_GIS_CHROMA_EXE" "missing chroma exe error names the override"
    }
    Assert-True $chromaExeFailed "missing explicit chroma exe must throw"
    [Environment]::SetEnvironmentVariable("HISTORICAL_GIS_CHROMA_EXE", $null)

    $env:HISTORICAL_GIS_CHROMA_DATA = Join-Path $explicitRoot "missing-chroma-data"
    $chromaDataFailed = $false
    try {
        Resolve-HistoricalGisLauncherConfig -SkipEmbeddingCache
    }
    catch {
        $chromaDataFailed = $true
        Assert-Contains $_.Exception.Message "HISTORICAL_GIS_CHROMA_DATA" "missing chroma data error names the override"
        Assert-NotContains $_.Exception.Message $legacyRoot "missing chroma data must not mention the original corpus as a substitute"
    }
    Assert-True $chromaDataFailed "missing explicit chroma data must throw"
    [Environment]::SetEnvironmentVariable("HISTORICAL_GIS_CHROMA_DATA", $null)

    $env:HISTORICAL_GIS_ENV_FILE = Join-Path $explicitRoot "missing.env"
    $envFailed = $false
    try {
        Resolve-HistoricalGisLauncherConfig -SkipEmbeddingCache
    }
    catch {
        $envFailed = $true
        Assert-Contains $_.Exception.Message "HISTORICAL_GIS_ENV_FILE" "missing env file error names the override"
    }
    Assert-True $envFailed "missing explicit env file must throw"
    [Environment]::SetEnvironmentVariable("HISTORICAL_GIS_ENV_FILE", $null)

    $env:ROMAN_ROAD_ENABLED = "true"
    $gisFailed = $false
    try {
        Assert-OptionalGisDependencies
    }
    catch {
        $gisFailed = $true
        Assert-Contains $_.Exception.Message "ROMAN_ROAD_GEOJSON_PATH" "enabled roman roads require an explicit geojson path"
    }
    Assert-True $gisFailed "enabled roman roads without a path must throw"
    [Environment]::SetEnvironmentVariable("ROMAN_ROAD_ENABLED", $null)

    $env:MARITIME_SURFACE_DATA_ROOT = Join-Path $tempRoot "missing-maritime"
    $maritimeFailed = $false
    try {
        Assert-OptionalGisDependencies
    }
    catch {
        $maritimeFailed = $true
        Assert-Contains $_.Exception.Message "MARITIME_SURFACE_DATA_ROOT" "missing maritime root must throw only when configured"
    }
    Assert-True $maritimeFailed "missing configured maritime root must throw"
    [Environment]::SetEnvironmentVariable("MARITIME_SURFACE_DATA_ROOT", $null)

    Assert-OptionalGisDependencies

    $incompatible = Resolve-HistoricalGisChromaPort 8002 $script:chromaData
    Assert-True ($incompatible.Status -in @("Free", "SameConfig", "IncompatibleRuntime", "UnknownOccupant", "UnverifiedChroma")) "chroma occupancy probe must return a known status"

    $sameChromaCommand = 'chroma.exe run --path "' + $explicitRoot + '\data\chroma_server_roman_republic_v2" --host 127.0.0.1 --port 8002'
    $otherChromaCommand = 'chroma.exe run --path "' + $legacyRoot + '\data\chroma_server_roman_republic_v2" --host 127.0.0.1 --port 8002'
    Assert-True ((Normalize-ProjectPath (Get-ChromaPathFromCommandLine $sameChromaCommand)) -eq (Normalize-ProjectPath (Join-Path $explicitRoot "data\chroma_server_roman_republic_v2"))) "chroma --path parsing"
    Assert-True ((Normalize-ProjectPath (Get-ChromaPathFromCommandLine $otherChromaCommand)) -ne (Normalize-ProjectPath (Join-Path $explicitRoot "data\chroma_server_roman_republic_v2"))) "incompatible chroma path must not match the configured root"

    Write-Host "launcher path resolution: PASS"
}
finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name])
    }
    $script:DefaultSharedRuntime = "C:\data\historical-gis-runtime"
    $script:LegacySharedRuntime = "C:\D\python\202608231533"
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
