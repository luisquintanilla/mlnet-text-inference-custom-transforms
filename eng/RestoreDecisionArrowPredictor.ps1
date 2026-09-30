param(
    [Parameter(Mandatory = $true)][string] $ArtifactRoot,
    [switch] $InitializeLock,
    [switch] $ProjectOnly
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$pins = Get-Content -LiteralPath "$PSScriptRoot\experiments\decision-arrow.dependencies.json" -Raw | ConvertFrom-Json
if ($pins.schemaVersion -ne 1) { throw 'Unsupported dependency receipt version.' }
$ArtifactRoot = (Resolve-Path -LiteralPath $ArtifactRoot).Path

foreach ($feed in @($pins.packages.feed | Sort-Object -Unique)) {
    $receipt = Get-Content -LiteralPath (Join-Path $ArtifactRoot "$feed\receipt.json") -Raw | ConvertFrom-Json
    $isArrow = $feed -eq 'feeds\arrow-baseline'
    $expectedCommit = if ($isArrow) { $pins.arrowCommit } else { $pins.adapterCommit }
    $expectedRepository = if ($isArrow) { 'https://github.com/luisquintanilla/arrow-dotnet' } else { 'https://github.com/luisquintanilla/typesafe-meai' }
    if ($receipt.schemaVersion -ne 1 -or $receipt.status -cne 'READY' -or
        $receipt.source.cleanTree -ne $true -or $receipt.source.commit -cne $expectedCommit -or
        $receipt.source.repository -cne $expectedRepository -or $receipt.packaging.immutable -ne $true) {
        throw "Producer source/immutable receipt mismatch for $feed."
    }
    foreach ($pin in @($pins.packages | Where-Object { $_.feed -eq $feed })) {
        $declared = @($receipt.packages | Where-Object { $_.id -ceq $pin.id -and $_.version -ceq $pin.version })
        if ($declared.Count -ne 1 -or $declared[0].sha256 -cne $pin.sha256 -or
            $declared[0].sizeBytes -ne $pin.sizeBytes -or $declared[0].file -cne $pin.file) {
            throw "Package receipt mismatch for $($pin.id)."
        }
        $path = Join-Path $ArtifactRoot "$feed\$($pin.file)"
        if ((Get-Item -LiteralPath $path).Length -ne $pin.sizeBytes -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $pin.sha256) {
            throw "Package byte/hash mismatch for $($pin.id)."
        }
    }
}

$consumer = Join-Path $ArtifactRoot 'predictors'
New-Item -ItemType Directory -Path $consumer -Force | Out-Null
$config = Join-Path $consumer 'pinned.NuGet.config'
$adapter = [System.Security.SecurityElement]::Escape((Join-Path $ArtifactRoot $pins.packages[0].feed))
$arrow = [System.Security.SecurityElement]::Escape((Join-Path $ArtifactRoot 'feeds\arrow-baseline'))
$cache = Join-Path $consumer 'packages-pinned-v1'
$escapedCache = [System.Security.SecurityElement]::Escape($cache)
@"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="decision-adapter" value="$adapter" />
    <add key="arrow-baseline" value="$arrow" />
    <add key="nuget.org" value="https://www.nuget.org/api/v2/" protocolVersion="2" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="decision-adapter">
      <package pattern="DecisionInference.Arrow" />
      <package pattern="DecisionInference.Abstractions" />
    </packageSource>
    <packageSource key="arrow-baseline">
      <package pattern="Apache.Arrow" />
      <package pattern="Apache.Arrow.Scalars" />
      <package pattern="Apache.Arrow.Compute" />
    </packageSource>
    <packageSource key="nuget.org"><package pattern="*" /></packageSource>
  </packageSourceMapping>
  <config>
    <add key="signatureValidationMode" value="accept" />
    <add key="globalPackagesFolder" value="$escapedCache" />
  </config>
</configuration>
"@ | Set-Content -LiteralPath $config -Encoding utf8NoBOM

$target = if ($ProjectOnly) { "$repo\samples\DecisionArrowPredictor\DecisionArrowPredictor.csproj" } else { "$repo\eng\DecisionArrowPredictor.slnx" }
$restoreArgs = @('restore', $target, '--configfile', $config,
    '--packages', $cache, '-p:ImportDirectoryBuildTargets=false', '--verbosity', 'quiet')
if (-not $InitializeLock) { $restoreArgs += '--locked-mode' }
& dotnet @restoreArgs
if ($LASTEXITCODE -ne 0) { throw "Predictor restore failed ($LASTEXITCODE)." }

$assets = Get-Content -LiteralPath "$repo\samples\DecisionArrowPredictor\obj\project.assets.json" -Raw | ConvertFrom-Json
foreach ($pin in $pins.packages) {
    if ($null -eq $assets.libraries.PSObject.Properties["$($pin.id)/$($pin.version)"]) {
        throw "Resolved package version differs from pinned artifact: $($pin.id)."
    }
}
Write-Output "Verified pinned restore; config=$config; isolatedCache=$cache"
