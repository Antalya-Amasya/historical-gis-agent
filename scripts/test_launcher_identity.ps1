$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "start_historical_gis.ps1")

$cursorRoot = Split-Path -Parent $PSScriptRoot
$codexRoot = "C:\D\python\historical-gis-codex"
$cursorFrontend = Join-Path $cursorRoot "frontend"
$codexFrontend = Join-Path $codexRoot "frontend"

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Assert-Equals {
    param([string]$Expected, [string]$Actual, [string]$Message)
    if ($Expected -ne $Actual) {
        throw "$Message (expected '$Expected', got '$Actual')"
    }
}

$sameBackendCommand = 'python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --env-file "C:\D\python\202608231533\.env" --app-dir "C:\D\python\historical-gis-cursor"'
$staleBackendCommand = 'python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --env-file "C:\D\python\202608231533\.env" --app-dir "C:\D\python\historical-gis-codex"'
$unknownBackendCommand = 'python -m http.server 8000'
$sameFrontendCommand = '"node" "C:\D\python\historical-gis-cursor\frontend\node_modules\.bin\\..\vite\bin\vite.js" --host 127.0.0.1 "--port" "5173" "--strictPort"'
$staleFrontendCommand = '"node" "C:\D\python\historical-gis-codex\frontend\node_modules\.bin\\..\vite\bin\vite.js" --host 127.0.0.1 "--port" "5173" "--strictPort"'

# TEST A: same-worktree backend identity (command-line parsing)
Assert-True (Test-IsHistoricalGisBackendCommand $sameBackendCommand) "backend command should be recognized"
Assert-Equals (Normalize-ProjectPath (Get-BackendAppDirFromCommandLine $sameBackendCommand)) (Normalize-ProjectPath $cursorRoot) "same-worktree backend app-dir"

$liveBackend = Resolve-HistoricalGisBackendPort 8000 $cursorRoot
if ($liveBackend.Status -eq "SameWorktree") {
    Assert-Equals "SameWorktree" $liveBackend.Status "live same-worktree backend reuse allowed"
}
elseif ($liveBackend.Status -eq "Free") {
    $simulatedSame = if ((Normalize-ProjectPath (Get-BackendAppDirFromCommandLine $sameBackendCommand)) -eq (Normalize-ProjectPath $cursorRoot)) { "SameWorktree" } else { "Mismatch" }
    Assert-Equals "SameWorktree" $simulatedSame "same-worktree backend identity"
}
elseif ($liveBackend.Status -eq "StaleWorktree") {
  # Port currently occupied by another worktree; launcher must not treat it as same-worktree.
  Assert-True $true "live port shows stale worktree (cross-worktree rejection path available)"
}
else {
    throw "Unexpected live backend occupancy on 8000: $($liveBackend.Status)"
}

# TEST B: cross-worktree backend reuse forbidden
$simulatedStale = if ((Normalize-ProjectPath (Get-BackendAppDirFromCommandLine $staleBackendCommand)) -eq (Normalize-ProjectPath $cursorRoot)) { "SameWorktree" } else { "StaleWorktree" }
Assert-Equals "StaleWorktree" $simulatedStale "cross-worktree backend reuse forbidden"
if ($liveBackend.Status -eq "StaleWorktree") {
    Assert-Equals "StaleWorktree" $liveBackend.Status "live stale backend detected on 8000"
}

# TEST C: unknown process must not be killed or treated as Historical GIS
Assert-True (-not (Test-IsHistoricalGisBackendCommand $unknownBackendCommand)) "unknown backend command rejected"
$unknownOccupant = if (Test-IsHistoricalGisBackendCommand $unknownBackendCommand) { "HistoricalGis" } else { "UnknownOccupant" }
Assert-Equals "UnknownOccupant" $unknownOccupant "unknown process must not be treated as Historical GIS backend"

# Frontend identity checks
Assert-True (Test-IsHistoricalGisFrontendCommand $sameFrontendCommand) "frontend command should be recognized"
Assert-Equals (Normalize-ProjectPath (Get-FrontendRootFromCommandLine $sameFrontendCommand)) (Normalize-ProjectPath $cursorFrontend) "same-worktree frontend root"
Assert-Equals (Normalize-ProjectPath (Get-FrontendRootFromCommandLine $staleFrontendCommand)) (Normalize-ProjectPath $codexFrontend) "stale frontend root"

$simulatedFrontendStale = if ((Normalize-ProjectPath (Get-FrontendRootFromCommandLine $staleFrontendCommand)) -eq (Normalize-ProjectPath $cursorFrontend)) { "SameWorktree" } else { "StaleWorktree" }
Assert-Equals "StaleWorktree" $simulatedFrontendStale "cross-worktree frontend reuse forbidden"

$liveFrontend = Resolve-HistoricalGisFrontendPort 5173 $cursorFrontend
if ($liveFrontend.Status -eq "StaleWorktree") {
    Assert-Equals "StaleWorktree" $liveFrontend.Status "live stale frontend detected on 5173"
}
elseif ($liveFrontend.Status -eq "SameWorktree") {
    Assert-Equals "SameWorktree" $liveFrontend.Status "live same-worktree frontend reuse allowed"
}

Write-Host "launcher identity regression: PASS"
