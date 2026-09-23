[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $BundlePath,

    [ValidateSet("facade", "stages")]
    [string] $Mode = "facade",

    [switch] $MLNet
)

$resolvedBundle = Resolve-Path -LiteralPath $BundlePath -ErrorAction Stop
if (-not (Test-Path -LiteralPath $resolvedBundle -PathType Container) -and
    -not ($resolvedBundle.Path.EndsWith(".zip", [StringComparison]::OrdinalIgnoreCase))) {
    throw "BundlePath must be a typed-decision bundle directory or .zip archive."
}

$required = @(
    "typed-decision-bundle.json"
)
if (Test-Path -LiteralPath $resolvedBundle -PathType Container) {
    foreach ($file in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $resolvedBundle $file) -PathType Leaf)) {
            throw "Bundle is missing '$file'. No model assets are downloaded by this script."
        }
    }
}

if ($MLNet) {
    & dotnet run --file (Join-Path $PSScriptRoot "..\..\samples\TypedDecisions\MLNetPipeline\Program.cs") `
        -- --mode $Mode --bundle $resolvedBundle.Path
}
else {
    & dotnet run --file (Join-Path $PSScriptRoot "..\..\samples\TypedDecisions\Standalone\Program.cs") `
        -- --mode $Mode --bundle $resolvedBundle.Path
}

if ($LASTEXITCODE -ne 0) {
    throw "Typed-decision acceptance execution failed with exit code $LASTEXITCODE."
}
