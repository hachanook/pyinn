# PyINN Docker Setup

This directory contains all Docker-related files for the PyINN project.

## Quick Start

### Option 1: Using Pre-built Image from Docker Hub (Recommended)

```bash
docker run chanwookpark2024/pyinn:latest
```

### Option 2: Build and Run Locally

**From the project root directory:**

**Linux/macOS:**
```bash
cd /path/to/pyinn
./docker/run_docker.sh
```

**Windows:**
```cmd
cd \path\to\pyinn
docker\run_docker.bat
```

**Manual build:**
```bash
# From project root
docker build -f docker/Dockerfile -t pyinn .
docker run pyinn
```

### Option 3: Using Docker Compose

```bash
# From project root
docker-compose -f docker/docker-compose.yml up --build
```

### Option 4: Running Jupyter Notebooks (Recommended for Tutorials)

**Quick Start:**

**Linux/macOS:**
```bash
./docker/run_jupyter.sh
```

**Windows:**
```cmd
docker\run_jupyter.bat
```

**Using Docker Compose:**
```bash
docker-compose -f docker/docker-compose-jupyter.yml up --build
```

**Manual Jupyter Launch:**
```bash
# From project root
docker run -it --rm -p 8888:8888 \
  -v $(pwd)/tutorials:/app/tutorials \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/plots:/app/plots \
  chanwookpark2024/pyinn:latest \
  conda run -n pyinn-env jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app
```

Then open your browser and go to `http://localhost:8888` to access the tutorials!

## Files in this Directory

- `Dockerfile` - Main Docker configuration
- `docker-compose.yml` - Docker Compose configuration for main app
- `docker-compose-jupyter.yml` - Docker Compose configuration for Jupyter
- `environment_docker.yaml` - Conda environment without PyTorch
- `.dockerignore` - Files to exclude from Docker build
- `run_docker.sh` - Linux/macOS build and run script
- `run_docker.bat` - Windows build and run script
- `run_jupyter.sh` - Linux/macOS Jupyter launch script
- `run_jupyter.bat` - Windows Jupyter launch script
- `push_to_dockerhub.sh` - Script to push to Docker Hub (Linux/macOS)
- `push_to_dockerhub.bat` - Script to push to Docker Hub (Windows)
- `DOCKER_README.md` - Detailed Docker documentation

## For Detailed Documentation

See `DOCKER_README.md` for comprehensive setup instructions, troubleshooting, and advanced usage.

## Docker Hub

The pre-built image is available at: https://hub.docker.com/r/chanwookpark2024/pyinn 