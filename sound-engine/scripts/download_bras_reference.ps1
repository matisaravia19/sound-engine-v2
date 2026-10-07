param([string]$Destination = "output/measured-reference-research")
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$manifest = Get-Content -LiteralPath (Join-Path $repo 'references/bras-downloads.json') -Raw | ConvertFrom-Json
$target = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $target | Out-Null
# Only the RIR archives and complete on-axis IR row are needed to reproduce
# this comparison. The larger directional response remains an explicit omission.
$required = @('BRAS_Documentation.pdf', 'BRAS_RS5.zip', 'BRAS_RS6.zip', 'genelec-ir-prefix.bin', 'material-selected.bin')
foreach ($entry in $manifest.files) {
    if ($entry.file -notin $required) { continue }
    $path = Join-Path $target $entry.file
    if (-not (Test-Path -LiteralPath $path)) {
        $headers = @{}
        if ($entry.range) { $headers.Range = "bytes=$($entry.range[0])-$($entry.range[1])" }
        Invoke-WebRequest -Uri $entry.url -Headers $headers -OutFile $path -TimeoutSec 180
    }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw "Hash mismatch for $path; inspect the download before using it."
    }
    Write-Output "Verified $($entry.file)"
}
