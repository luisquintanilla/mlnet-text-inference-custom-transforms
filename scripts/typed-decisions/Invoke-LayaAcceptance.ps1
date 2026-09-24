[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $ModelAssetsPath,

    [ValidateSet(
        "direct",
        "facade",
        "stages",
        "composed",
        "prediction-engine",
        "prediction-engine-stages",
        "prediction-engine-composed")]
    [string] $Mode = "facade"
)

$resolvedAssets = Resolve-Path -LiteralPath $ModelAssetsPath -ErrorAction Stop
if (-not (Test-Path -LiteralPath $resolvedAssets -PathType Container) -and
    -not ($resolvedAssets.Path.EndsWith(".zip", [StringComparison]::OrdinalIgnoreCase))) {
    throw "ModelAssetsPath must be a model-assets directory or .zip archive."
}

& dotnet run --file (Join-Path $PSScriptRoot "..\..\samples\TypedDecisions\MLNetPipeline\Program.cs") `
    -- --mode $Mode --model-assets $resolvedAssets.Path

if ($LASTEXITCODE -ne 0) {
    throw "Typed-decision acceptance execution failed with exit code $LASTEXITCODE."
}
