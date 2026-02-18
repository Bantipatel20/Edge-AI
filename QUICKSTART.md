# 🚀 Quick Start Guide: Deploy to Jetson Nano

## ✅ Windows Setup Completed

Your Windows environment is now configured with:
- ✅ Docker Desktop installed
- ✅ Buildx enabled for multi-architecture
- ✅ ARM64 platform support verified
- ✅ Dockerfile created and optimized
- ✅ Build scripts ready

---

## 📦 Two Deployment Options

### Option 1: Build on Windows → Push to Docker Hub → Pull on Jetson (Slower but convenient)

**Pros:** Build once, deploy anywhere  
**Cons:** Cross-platform builds are slow (10-15 min), may timeout on large dependencies

```powershell
# On Windows
.\docker-build.ps1 -dockerhub_username "YOUR_USERNAME" -push
# Will build and push directly to Docker Hub
```

Then on Jetson Nano:
```bash
docker pull YOUR_USERNAME/edge-ai-webapp:latest
docker run -d -p 8501:8501 --name edge-ai YOUR_USERNAME/edge-ai-webapp:latest
```

### Option 2: Build Directly on Jetson Nano (Recommended ✨)

**Pros:** Faster, no cross-compilation, more reliable  
**Cons:** Requires git or file transfer to Jetson

#### Step 1: Transfer Files to Jetson
```bash
# From Windows (if you have git on Jetson)
# Already done if you clone the repo on Jetson

# OR use SCP to transfer
scp -r C:\Users\HP\AppData\Local\GitHubDesktop\Edge-AI jetson@<jetson-ip>:~/
```

#### Step 2: Build on Jetson
```bash
# SSH into Jetson Nano
ssh jetson@<jetson-ip>

# Navigate to project
cd ~/Edge-AI

# Build (uses native ARM64, much faster)
docker build -t edge-ai-webapp:latest .

# Run
docker run -d \
  -p 8501:8501 \
  --name edge-ai \
  --restart unless-stopped \
  edge-ai-webapp:latest

# Check status
docker ps
docker logs edge-ai
```

#### Step 3: Access Application
Open browser: `http://<jetson-ip>:8501`

---

## 📋 Files Created

| File | Purpose |
|------|---------|
| `Dockerfile` | Docker image definition for Streamlit app |
| `.dockerignore` | Excludes unnecessary files from image |
| `docker-build.ps1` | Automated build script (Windows) |
| `setup-docker.ps1` | Verify Docker setup (Windows) |
| `docker-compose.yml` | Easy deployment configuration (Jetson) |
| `DOCKER_DEPLOYMENT_GUIDE.md` | Complete reference guide |

---

## 🔄 Quick Commands Reference

### Windows Commands
```powershell
# Verify setup
.\setup-docker.ps1

# Build only (test)
.\docker-build.ps1 -dockerhub_username "YOUR_USERNAME"

# Build and push
.\docker-build.ps1 -dockerhub_username "YOUR_USERNAME" -push

# Check images
docker images
```

### Jetson Nano Commands
```bash
# Using Docker Compose (easiest)
docker-compose up -d
docker-compose logs -f
docker-compose down

# Using Docker directly
docker run -d -p 8501:8501 --name edge-ai edge-ai-webapp:latest
docker logs -f edge-ai
docker stop edge-ai
docker rm edge-ai

# System info
docker stats edge-ai
docker inspect edge-ai
```

---

## 🐛 Known Issues & Solutions

### Issue: Windows build times out
**Solution:** Use Option 2 (build on Jetson directly)

### Issue: "Out of memory" on Jetson
**Solution:** Enable swap
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: Port 8501 already in use
**Solution:** 
```bash
# Find and stop conflicting process
sudo lsof -i :8501
# Or use different port
docker run -p 8080:8501 ...
```

---

## 📊 What to Expect

### Build Time
- **Windows (cross-platform):** 10-15 minutes
- **Jetson Nano (native):** 5-8 minutes

### Image Size
- **Compressed:** ~800MB
- **Uncompressed:** ~2GB (PyTorch is large)

### Runtime Resources
- **RAM:** ~500MB-1GB
- **CPU:** 20-40% (during inference)

---

## ✅ Success Checklist

After deployment, verify:
- [ ] Container is running: `docker ps` shows "edge-ai"
- [ ] Logs show no errors: `docker logs edge-ai`
- [ ] Port is accessible: Browser opens `http://<ip>:8501`
- [ ] App loads: You see "Image Classification" interface
- [ ] Upload works: Can upload and classify images

---

## 🎯 Recommended Workflow

**For Development:**
```powershell
# On Windows - make code changes
# Test locally if possible
# Push to GitHub

# On Jetson - pull and rebuild
git pull
docker build -t edge-ai-webapp:latest .
docker stop edge-ai && docker rm edge-ai
docker run -d -p 8501:8501 --name edge-ai --restart unless-stopped edge-ai-webapp:latest
```

**For Production:**
```powershell
# On Windows - build and push to Docker Hub
.\docker-build.ps1 -dockerhub_username "YOUR_USERNAME" -push

# On Jetson - pull and run
docker pull YOUR_USERNAME/edge-ai-webapp:latest
docker-compose up -d
```

---

## 📚 Additional Resources

- Full guide: [DOCKER_DEPLOYMENT_GUIDE.md](DOCKER_DEPLOYMENT_GUIDE.md)
- Docker docs: https://docs.docker.com
- Jetson setup: https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

---

**Need help?** Check the troubleshooting section in `DOCKER_DEPLOYMENT_GUIDE.md`
