param(
    [Parameter(Mandatory = $true)][string] $AssetsFile,
    [Parameter(Mandatory = $true)][string] $DepsFile
)

$ErrorActionPreference = 'Stop'
$assets = Get-Content -LiteralPath $AssetsFile -Raw | ConvertFrom-Json
$deps = Get-Content -LiteralPath $DepsFile -Raw | ConvertFrom-Json
$forbidden = '(?i)onnx|cuda|tokeniz|laya|tensorsharp|mlnet\.textinference'
$libraries = @($assets.libraries.PSObject.Properties.Name) + @($deps.libraries.PSObject.Properties.Name)
$paths = @(
    foreach ($target in $deps.targets.PSObject.Properties) {
        foreach ($library in $target.Value.PSObject.Properties) {
            foreach ($kind in @('runtime', 'native', 'runtimeTargets')) {
                $item = $library.Value.PSObject.Properties[$kind]
                if ($null -ne $item) { $item.Value.PSObject.Properties.Name }
            }
        }
    }
)
$leaks = @(($libraries + $paths) | Where-Object { $_ -match $forbidden } | Sort-Object -Unique)
if ($leaks.Count -gt 0) {
    throw "Predictor dependency closure contains forbidden inference dependencies: $($leaks -join ', ')"
}
$ml = @($assets.libraries.PSObject.Properties.Name | Where-Object { $_ -eq 'Microsoft.ML/5.0.0' })
if ($ml.Count -ne 1) {
    throw 'Expected resolved Microsoft.ML exactly 5.0.0.'
}
Write-Output 'Predictor resolved build/runtime closure: CPU-only, no ONNX/CUDA/tokenizer/provider dependency.'
