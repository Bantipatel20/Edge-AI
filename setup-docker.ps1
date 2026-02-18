# Quick Setup Script for Docker ARM64 Build
# Run this script to set up and test your Docker environment

Write-Host "=== Docker Setup for Jetson Nano Deployment ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker
Write-Host "Step 1: Checking Docker installation..." -ForegroundColor Yellow
$dockerVersion = docker --version
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ $dockerVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Step 2: Check Buildx
Write-Host "`nStep 2: Checking Docker Buildx..." -ForegroundColor Yellow
$buildxVersion = docker buildx version
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Buildx available" -ForegroundColor Green
} else {
    Write-Host "❌ Buildx not found." -ForegroundColor Red
    exit 1
}

# Step 3: Setup builder
Write-Host "`nStep 3: Setting up ARM64 builder..." -ForegroundColor Yellow
docker buildx create --name arm-builder --use 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Builder already exists, using existing one..." -ForegroundColor Cyan
    docker buildx use arm-builder
}

# Step 4: Inspect builder
Write-Host "`nStep 4: Inspecting builder capabilities..." -ForegroundColor Yellow
docker buildx inspect arm-builder | Select-String "linux/arm64"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ ARM64 platform support confirmed" -ForegroundColor Green
} else {
    Write-Host "⚠️  ARM64 support unclear, but proceeding..." -ForegroundColor Yellow
}

# Step 5: List builders
Write-Host "`nStep 5: Available builders:" -ForegroundColor Yellow
docker buildx ls

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit docker-build.ps1 and set your Docker Hub username" -ForegroundColor White
Write-Host "2. Run: .\docker-build.ps1 -dockerhub_username 'your-username'" -ForegroundColor White
Write-Host ""
Write-Host "Note: ARM64 builds can take 10-15 minutes due to:" -ForegroundColor Yellow
Write-Host "  - Cross-platform emulation" -ForegroundColor White
Write-Host "  - Large PyTorch dependency (~90MB)" -ForegroundColor White
Write-Host "  - System libraries compilation" -ForegroundColor White
Write-Host ""
Write-Host "Alternative: Build directly on Jetson Nano (faster)" -ForegroundColor Cyan
Write-Host "  1. Copy project to Jetson via git or scp" -ForegroundColor White
Write-Host "  2. On Jetson: docker build -t edge-ai:latest ." -ForegroundColor White
Write-Host "  3. On Jetson: docker run -d -p 8501:8501 edge-ai:latest" -ForegroundColor White
