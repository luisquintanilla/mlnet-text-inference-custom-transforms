param(
    [string] $ArtifactRoot,
    [switch] $Help
)

$ErrorActionPreference = 'Stop'
if ($Help) {
    Write-Output @'
Start the DecisionArrowPredictor offline walkthrough.
From the repository root:
  .\samples\DecisionArrowPredictor\Start.ps1 -ArtifactRoot <local artifact-kit folder>
The kit supplies pinned experimental packages and receipts, not models or SMS data.
This command verifies/restores/builds the sample, then reads a synthetic Arrow fixture.
It does not run Laya, acquire a corpus, train a classifier or report accuracy.
'@
    return
}

if ([string]::IsNullOrWhiteSpace($ArtifactRoot) -or
    -not (Test-Path -LiteralPath $ArtifactRoot -PathType Container)) {
    throw 'Supply -ArtifactRoot with the existing producer-supplied decision-to-Arrow artifact kit. It must contain the pinned package feeds and receipts; a corpus/model folder is not a kit. See samples\DecisionArrowPredictor\README.md.'
}
$ArtifactRoot = (Resolve-Path -LiteralPath $ArtifactRoot).Path
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pins = Get-Content -LiteralPath "$repo\eng\experiments\decision-arrow.dependencies.json" -Raw | ConvertFrom-Json
$required = @(
    foreach ($pin in $pins.packages) {
        Join-Path $ArtifactRoot "$($pin.feed)\receipt.json"
        Join-Path $ArtifactRoot "$($pin.feed)\$($pin.file)"
    }
) | Sort-Object -Unique
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -gt 0) {
    throw "Artifact kit is incomplete. Obtain the pinned package/receipt kit from the experiment producer. Missing: $($missing -join '; '). See samples\DecisionArrowPredictor\EXPERIMENT.md#artifact-kit-and-offline-start."
}

Write-Output '1/3 Verify the artifact kit and restore the sample (no model or corpus acquisition).'
& "$repo\eng\RestoreDecisionArrowPredictor.ps1" -ArtifactRoot $ArtifactRoot -ProjectOnly | Out-Null
Write-Output '2/3 Build the CPU-only table consumer.'
& dotnet build "$PSScriptRoot\DecisionArrowPredictor.csproj" --no-restore -p:ImportDirectoryBuildTargets=false --verbosity quiet
if ($LASTEXITCODE -ne 0) { throw "Sample build failed ($LASTEXITCODE); no smoke result produced." }
Write-Output '3/3 Read the checked-in synthetic Arrow fixture.'
& dotnet "$PSScriptRoot\bin\Debug\net10.0\DecisionArrowPredictor.dll" smoke
if ($LASTEXITCODE -ne 0) { throw "Synthetic Arrow smoke failed ($LASTEXITCODE)." }
Write-Output 'Read 257 synthetic rows as ten numeric features plus a separate direct baseline.'
Write-Output 'No classifier was trained; no accuracy or generalization result was measured.'
Write-Output 'Next: follow the ML.NET learning code in samples\DecisionArrowPredictor\README.md.'
