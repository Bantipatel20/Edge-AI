# Docker Build Script for Windows PowerShell
# Builds ARM64 image for Jetson Nano

param(
    [string]$dockerhub_username = "banti2010",
    [string]$image_name = "edge-ai-webapp",
    [string]$tag = "latest",
    [switch]$push = $false
)

$full_image = "${dockerhub_username}/${image_name}:${tag}"

Write-Host "🚀 Building ARM64 Docker Image: $full_image" -ForegroundColor Cyan
Write-Host "⏱️  This may take 10-15 minutes (PyTorch is large)..." -ForegroundColor Yellow
Write-Host ""

# Step 1: Create and use buildx builder
Write-Host "📦 Setting up Docker Buildx..." -ForegroundColor Yellow
docker buildx create --name arm-builder --use 2>$null
if ($LASTEXITCODE -ne 0) {
    docker buildx use arm-builder
}

# Step 2: Build for ARM64 platform
Write-Host "`n🔨 Building for linux/arm64..." -ForegroundColor Yellow
Write-Host "Progress will be shown below..." -ForegroundColor Cyan
Write-Host ""

if ($push) {
    # Build and push directly (faster, no local load)
    docker buildx build `
        --platform linux/arm64 `
        -t $full_image `
        --push `
        .
} else {
    # Build only (for testing)
    docker buildx build `
        --platform linux/arm64 `
        -t $full_image `
        --load `
        .
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Build successful!" -ForegroundColor Green
    Write-Host "`n📤 To push to Docker Hub, run:" -ForegroundColor Cyan
    Write-Host "   docker login" -ForegroundColor White
    Write-Host "   docker push $full_image" -ForegroundColor White
    Write-Host "`n🧲 To pull on Jetson Nano, run:" -ForegroundColor Cyan
    Write-Host "   docker pull $full_image" -ForegroundColor White
    Write-Host "   docker run -d -p 8501:8501 --name edge-ai $full_image" -ForegroundColor White
} else {
    Write-Host "`n❌ Build failed!" -ForegroundColor Red
    exit 1
}
