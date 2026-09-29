param(
    [Parameter(Mandatory = $true)][string] $ArtifactRoot
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$receipt = Get-Content -LiteralPath "$ArtifactRoot\contracts\decision-arrow\fixture-receipt.ab4eebf.json" -Raw | ConvertFrom-Json
if ($receipt.status -cne 'READY' -or $receipt.sourceCommit -cne 'ab4eebf29b9fdca49ce13e3b4afac0ec16b22983' -or
    $receipt.fixtureScope -cne 'synthetic-observations-not-real-inference' -or
    $receipt.questionsSha256 -cne '9e6e6c5c580934309cb1f9b4572c7b87a93ce22b32b50cb8b3af20caa20068ea') {
    throw 'Expected immutable clean-source synthetic fixture receipt.'
}
$expected = @{
    HighPrecision = '06654530d526e33db86ca9f59e937d79dda9d83f1447d675edb69856a9017513'
    FourDecimalPlaces = '66e0d62923ab097c24016eb711bd65d3d07a5f3a77fef8bec3b95e735fc8a2c1'
    TwoDecimalPlaces = 'b717b6c1c8250773be6d5594395405a81c64bae64d416d1c029a5e8ec3ed41e4'
}
if ($receipt.fixtures.Count -ne 3) { throw 'Expected all three precision fixtures.' }
foreach ($fixture in $receipt.fixtures) {
    if ($fixture.fingerprint -cne $expected[$fixture.precision] -or $fixture.rows -ne 257) {
        throw 'Fixture identity/count mismatch.'
    }
    $destination = "$root\samples\DecisionArrowPredictor\fixtures\$($fixture.precision)"
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($item in @(
        @{ File = 'contract.json'; Hash = $fixture.contractSha256 },
        @{ File = 'manifest.json'; Hash = $fixture.manifestSha256 },
        @{ File = 'decisions.arrow'; Hash = $fixture.dataSha256 }
    )) {
        $source = Join-Path $fixture.directory $item.File
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne $item.Hash) {
            throw "Fixture hash mismatch: $source"
        }
        $target = Join-Path $destination $item.File
        if (Test-Path -LiteralPath $target) {
            if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -cne $item.Hash) {
                throw "Existing fixture differs: $target"
            }
        } else {
            Copy-Item -LiteralPath $source -Destination $target
        }
    }
}
Write-Output 'Imported exact clean-source synthetic fixtures; no corpus text or model assets.'
