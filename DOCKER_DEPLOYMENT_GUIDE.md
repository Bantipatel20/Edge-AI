# 🚀 Docker Deployment Guide: Windows → Jetson Nano

## 📋 Architecture Overview

```
Windows Laptop
   ├── Project Code
   ├── Docker Build (ARM64)
   └── Push Image → Docker Hub

GitHub → Stores Source Code

Jetson Nano
   └── Pull Image → Run Container → Website (Port 8501)
```

---

## 🔧 Step 1: Install Docker Desktop on Windows

1. Download & install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Enable in Settings:
   - ✅ WSL2 backend
   - ✅ Experimental features
3. Verify installation:
   ```powershell
   docker --version
   docker buildx version
   ```

---

## 🔧 Step 2: Enable Buildx (Multi-Architecture)

```powershell
docker buildx create --name arm-builder --use
docker buildx inspect --bootstrap
```

Verify ARM64 support:
```powershell
docker buildx ls
```

You should see `linux/arm64` in the platforms list.

---

## 🧱 Step 3: Build ARM64 Image

### Option A: Using PowerShell Script (Recommended)

```powershell
# Edit docker-build.ps1 and set your Docker Hub username
.\docker-build.ps1 -dockerhub_username "your-dockerhub-username"
```

### Option B: Manual Build

```powershell
docker buildx build --platform linux/arm64 -t your-dockerhub-username/edge-ai-webapp:latest --load .
```

**Note:** Replace `your-dockerhub-username` with your actual Docker Hub username.

---

## 🔐 Step 4: Push to Docker Hub

```powershell
# Login to Docker Hub
docker login

# Push the image
docker push your-dockerhub-username/edge-ai-webapp:latest
```

---

## 🧲 Step 5: Pull & Run on Jetson Nano

### On Jetson Nano (via SSH or direct terminal):

```bash
# Pull the image
docker pull your-dockerhub-username/edge-ai-webapp:latest

# Run the container
docker run -d \
  -p 8501:8501 \
  --name edge-ai \
  --restart unless-stopped \
  your-dockerhub-username/edge-ai-webapp:latest

# Check status
docker ps

# View logs
docker logs edge-ai
```

### Access the Application:
- **From Jetson Nano**: http://localhost:8501
- **From other devices on network**: http://<jetson-nano-ip>:8501

---

## 🔍 Useful Commands

### On Windows Laptop:
```powershell
# View local images
docker images

# Test image locally (if you have ARM emulation)
docker run -p 8501:8501 your-dockerhub-username/edge-ai-webapp:latest

# Remove builder
docker buildx rm arm-builder
```

### On Jetson Nano:
```bash
# Stop container
docker stop edge-ai

# Start container
docker start edge-ai

# Remove container
docker rm -f edge-ai

# View resource usage
docker stats edge-ai

# Shell into container (debugging)
docker exec -it edge-ai /bin/bash
```

---

## 📦 What's Included in Image

✅ Python 3.8  
✅ Streamlit web framework  
✅ PyTorch & TorchVision  
✅ OpenCV (for image processing)  
✅ Trained DenseNet model (`best_densenet_9ch.pth`)  
✅ Application code (`app.py`)  

❌ Training scripts (excluded)  
❌ Datasets (excluded)  
❌ Development artifacts (excluded)  

---

## 🐛 Troubleshooting

### Build fails on Windows:
- Ensure WSL2 is installed: `wsl --install`
- Enable "Use the WSL 2 based engine" in Docker Desktop settings
- Restart Docker Desktop

### Image won't run on Jetson Nano:
- Verify architecture: `docker inspect your-dockerhub-username/edge-ai-webapp:latest | grep Architecture`
- Should show: `"Architecture": "arm64"`

### App not accessible from browser:
- Check if container is running: `docker ps`
- Check logs: `docker logs edge-ai`
- Verify port is open: `sudo netstat -tulpn | grep 8501`
- Check firewall settings on Jetson Nano

### Out of memory on Jetson Nano:
- Reduce PyTorch threads: Add `ENV OMP_NUM_THREADS=2` to Dockerfile
- Enable swap: 
  ```bash
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

---

## 🎯 Quick Reference

| Action | Command |
|--------|---------|
| Build | `.\docker-build.ps1 -dockerhub_username "yourname"` |
| Push | `docker push yourname/edge-ai-webapp:latest` |
| Pull (Jetson) | `docker pull yourname/edge-ai-webapp:latest` |
| Run (Jetson) | `docker run -d -p 8501:8501 --name edge-ai yourname/edge-ai-webapp:latest` |
| Logs | `docker logs -f edge-ai` |
| Stop | `docker stop edge-ai` |
| Remove | `docker rm -f edge-ai` |

---

## 📝 Notes

- **Image Size**: ~1.5-2GB (PyTorch is large)
- **First Pull**: May take 10-30 minutes on Jetson Nano (slow internet)
- **RAM Usage**: ~500MB-1GB when running
- **Auto-restart**: Container will auto-restart on Jetson reboot (if using `--restart unless-stopped`)

---

## 🔄 Update Workflow

When you update your code:

1. **On Windows**: Rebuild and push
   ```powershell
   .\docker-build.ps1 -dockerhub_username "yourname"
   docker push yourname/edge-ai-webapp:latest
   ```

2. **On Jetson Nano**: Pull and restart
   ```bash
   docker pull yourname/edge-ai-webapp:latest
   docker stop edge-ai
   docker rm edge-ai
   docker run -d -p 8501:8501 --name edge-ai --restart unless-stopped yourname/edge-ai-webapp:latest
   ```

---

## ✅ Success Indicators

You'll know it's working when:
- ✅ Build completes without errors
- ✅ Push to Docker Hub succeeds
- ✅ `docker ps` shows container running on Jetson Nano
- ✅ You can access http://<jetson-ip>:8501 from browser
- ✅ Log shows: `You can now view your Streamlit app in your browser`

---

**Built with ❤️ for Edge-AI Deployment**
