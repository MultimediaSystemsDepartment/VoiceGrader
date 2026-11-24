param(
    [string]$OutputPath = (Join-Path (Split-Path $PSScriptRoot -Parent) "gpu-stress-tester.zip")
)

$ErrorActionPreference = "Stop"
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("gpu-stress-tester-" + [guid]::NewGuid())
$packageRoot = Join-Path $stagingRoot "gpu-stress-tester"
$excluded = @("data", "outputs", "model-cache", ".pytest_cache", "__pycache__", ".env")

try {
    New-Item -ItemType Directory -Path $packageRoot | Out-Null
    Get-ChildItem -LiteralPath $PSScriptRoot -Force |
        Where-Object { $_.Name -notin $excluded } |
        Copy-Item -Destination $packageRoot -Recurse -Force
    Get-ChildItem -LiteralPath $packageRoot -Directory -Recurse -Force |
        Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") } |
        Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath $packageRoot -File -Recurse -Force |
        Where-Object { $_.Extension -eq ".pyc" } |
        Remove-Item -Force
    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    Compress-Archive -Path $packageRoot -DestinationPath $OutputPath -CompressionLevel Optimal
    Write-Host "Created $OutputPath"
}
finally {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
}
