param(
    [Parameter(Mandatory = $true)][string] $ArtifactRoot
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pins = Get-Content -LiteralPath "$PSScriptRoot\experiments\decision-arrow.fixtures.json" -Raw | ConvertFrom-Json
$receipt = Get-Content -LiteralPath "$ArtifactRoot\contracts\decision-arrow\fixture-receipt.ab4eebf.json" -Raw | ConvertFrom-Json
if ($pins.schemaVersion -ne 1 -or $receipt.status -cne 'READY' -or $receipt.sourceCommit -cne $pins.sourceCommit -or
    $receipt.fixtureScope -cne 'synthetic-observations-not-real-inference' -or
    $receipt.questionsSha256 -cne '9e6e6c5c580934309cb1f9b4572c7b87a93ce22b32b50cb8b3af20caa20068ea' -or
    $receipt.packageReceiptSha256 -cne $pins.packageReceiptSha256) {
    throw 'Expected immutable clean-source synthetic fixture receipt.'
}
$packageReceipt = "$ArtifactRoot\feeds\decision-adapter\0.1.0-exp.decisions.1.gab4eebf29b9f\receipt.json"
if ((Get-FileHash -LiteralPath $packageReceipt -Algorithm SHA256).Hash.ToLowerInvariant() -cne $pins.packageReceiptSha256) {
    throw 'Fixture package receipt bytes changed.'
}
if ($receipt.fixtures.Count -ne 3 -or
    @($receipt.fixtures.precision | Sort-Object -Unique).Count -ne 3) {
    throw 'Expected all three distinct precision fixtures.'
}
foreach ($fixture in $receipt.fixtures) {
    $expected = @($pins.fixtures | Where-Object { $_.precision -ceq $fixture.precision })
    if ($expected.Count -ne 1) { throw 'Unknown fixture precision.' }
    $expected = $expected[0]
    if ($fixture.fingerprint -cne $expected.fingerprint -or $fixture.rows -ne $expected.rows -or
        $fixture.dataSize -ne $expected.dataSize -or $fixture.dataSha256 -cne $expected.dataSha256 -or
        $fixture.manifestSha256 -cne $expected.manifestSha256 -or $fixture.contractSha256 -cne $expected.contractSha256) {
        throw 'Fixture identity/count mismatch.'
    }
    $destination = "$root\samples\DecisionArrowPredictor\fixtures\$($fixture.precision)"
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($item in @(
        @{ File = 'contract.json'; Hash = $fixture.contractSha256 },
        @{ File = 'manifest.json'; Hash = $fixture.manifestSha256 },
        @{ File = 'decisions.arrow'; Hash = $fixture.dataSha256 }
    )) {
        $source = Join-Path "$ArtifactRoot\contracts\decision-arrow\$($expected.directory)" $item.File
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
