# PowerShell script to add Windows Defender exclusions for Python virtual environments
# This script must be run as Administrator

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To run as Administrator:" -ForegroundColor Yellow
    Write-Host "1. Right-click on PowerShell" -ForegroundColor Yellow
    Write-Host "2. Select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host "3. Navigate to this directory and run the script again" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or run this command in an Admin PowerShell:" -ForegroundColor Cyan
    Write-Host "Set-ExecutionPolicy Bypass -Scope Process -Force; & '$PSCommandPath'" -ForegroundColor Cyan
    pause
    exit 1
}

Write-Host "Adding Windows Defender exclusions..." -ForegroundColor Green
Write-Host ""

# Paths to exclude
$exclusions = @(
    "C:\Users\ValterAdmin\.venvs",
    "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis",
    "C:\code\econ_venv"
)

$successCount = 0
$failCount = 0

foreach ($path in $exclusions) {
    Write-Host "Adding exclusion for: $path" -ForegroundColor Cyan

    if (Test-Path $path) {
        try {
            Add-MpPreference -ExclusionPath $path -ErrorAction Stop
            Write-Host "  ✓ Successfully added exclusion" -ForegroundColor Green
            $successCount++
        }
        catch {
            Write-Host "  ✗ Failed to add exclusion: $_" -ForegroundColor Red
            $failCount++
        }
    }
    else {
        Write-Host "  ⚠ Path does not exist, but exclusion added anyway" -ForegroundColor Yellow
        try {
            Add-MpPreference -ExclusionPath $path -ErrorAction Stop
            $successCount++
        }
        catch {
            Write-Host "  ✗ Failed to add exclusion: $_" -ForegroundColor Red
            $failCount++
        }
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor White
Write-Host "Summary:" -ForegroundColor White
Write-Host "  Successful: $successCount" -ForegroundColor Green
Write-Host "  Failed: $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })
Write-Host "========================================" -ForegroundColor White
Write-Host ""

if ($successCount -gt 0) {
    Write-Host "Exclusions have been added to Windows Defender." -ForegroundColor Green
    Write-Host "You should now be able to run your Python scripts without issues." -ForegroundColor Green
    Write-Host ""
    Write-Host "To verify, run your script again:" -ForegroundColor Cyan
    Write-Host "  python python/stata_inputs/full_period_export.py" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
