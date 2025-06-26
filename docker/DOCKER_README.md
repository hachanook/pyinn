# PyINN Docker Setup

This document explains how to run the PyINN (Interpolating Neural Network) project using Docker.

## Prerequisites

1. **Docker**: Install Docker on your system
   - Windows: [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/install/)
   - macOS: [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)
   - Linux: [Docker Engine](https://docs.docker.com/engine/install/)

**Note**: This Docker setup is configured for CPU-only execution and is compatible with all systems including Windows, macOS, and Linux. No GPU or NVIDIA Docker is required.

## Fixing Docker Permission Issues (Linux)

If you get a "permission denied" error when trying to connect to the Docker daemon socket on Linux, run these commands:

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Apply the group change (or logout and login again)
newgrp docker

# Alternatively, you can run Docker commands with sudo
sudo docker --version
```

After running these commands, you should be able to use Docker without `sudo`.

## Quick Start

### Option 0: Using Pre-built Image from Docker Hub (Recommended)

If available, you can directly run the pre-built image:

```bash
# Pull and run the latest image from Docker Hub
docker run chanwookpark2024/pyinn:latest
```

### Option 1: Using the provided script

**Linux/macOS:**
```bash
# Make the script executable (if not already done)
chmod +x run_docker.sh

# Run the container
./run_docker.sh
```

**Windows:**
```cmd
# Run the batch script
run_docker.bat
```

### Option 2: Using Docker Compose

```bash
# Build and run with docker-compose
docker-compose up --build
```

### Option 3: Manual Docker commands

**Linux/macOS:**
```bash
# Build the image
docker build -t pyinn .

# Run the container
docker run \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/plots:/app/plots \
    -v $(pwd)/config:/app/config \
    pyinn
```

**Windows (Command Prompt):**
```cmd
# Build the image
docker build -t pyinn .

# Run the container
docker run -v %cd%/data:/app/data -v %cd%/plots:/app/plots -v %cd%/config:/app/config pyinn
```

**Windows (PowerShell):**
```powershell
# Build the image
docker build -t pyinn .

# Run the container
docker run -v ${PWD}/data:/app/data -v ${PWD}/plots:/app/plots -v ${PWD}/config:/app/config pyinn
```

## Configuration

### CPU Configuration

The Dockerfile is configured for CPU-only execution, making it compatible with all systems. JAX will automatically use available CPU cores for computation.

### Data and Output Directories

The container mounts the following directories:
- `./data` → `/app/data` (input data)
- `./plots` → `/app/plots` (output plots)
- `./config` → `/app/config` (configuration files)

### Running Different Experiments

To run different experiments, modify the configuration in `pyinn/settings.yaml` before building the container, or mount a custom configuration file.

## Troubleshooting

### Docker Permission Issues (Linux)

If you get "permission denied" errors:

1. Add your user to the docker group:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. Or run commands with sudo:
   ```bash
   sudo docker build -t pyinn .
   sudo docker run pyinn
   ```

### Memory Issues

If you encounter memory issues:

1. Increase Docker memory limits in Docker Desktop settings (Windows/macOS)
2. For Linux, ensure you have sufficient system memory available

### Build Issues

If the build fails:

1. Ensure you have sufficient disk space
2. Check your internet connection (needed for downloading dependencies)
3. Try building without cache:
   ```bash
   docker build --no-cache -t pyinn .
   ```

## Development

### Interactive Mode

To run the container in interactive mode for development:

**Linux/macOS:**
```bash
docker run -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/plots:/app/plots \
    -v $(pwd)/config:/app/config \
    -v $(pwd)/pyinn:/app/pyinn \
    pyinn /bin/bash
```

**Windows (Command Prompt):**
```cmd
docker run -it -v %cd%/data:/app/data -v %cd%/plots:/app/plots -v %cd%/config:/app/config -v %cd%/pyinn:/app/pyinn pyinn /bin/bash
```

### Jupyter Notebooks

To run Jupyter notebooks in the container:

**Linux/macOS:**
```bash
docker run -it \
    -v $(pwd)/tutorials:/app/tutorials \
    -p 8888:8888 \
    pyinn conda run -n pyinn-env jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Windows:**
```cmd
docker run -it -v %cd%/tutorials:/app/tutorials -p 8888:8888 pyinn conda run -n pyinn-env jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then access Jupyter at `http://localhost:8888`

## Performance Tips

1. **CPU Cores**: JAX will automatically use all available CPU cores for parallel computation
2. **Volume Mounting**: Mount only necessary directories to avoid performance overhead
3. **Memory**: Allocate sufficient memory to Docker for large datasets (especially important for CPU-only execution)
4. **Storage**: Use SSD storage for better I/O performance
5. **Docker Resources**: On Windows/macOS, increase Docker Desktop resource limits if needed

## Publishing to Docker Hub

### For Project Maintainers

To push the image to Docker Hub for public use:

1. **Edit the push script**:
   ```bash
   # Edit push_to_dockerhub.sh and replace YOUR_DOCKERHUB_USERNAME with your actual username
   nano push_to_dockerhub.sh
   ```

2. **Login to Docker Hub**:
   ```bash
   docker login
   ```

3. **Push the image**:
   ```bash
   ./push_to_dockerhub.sh
   ```

### Manual Push Commands

Alternatively, you can push manually:

```bash
# Replace YOUR_DOCKERHUB_USERNAME with your actual Docker Hub username
DOCKERHUB_USERNAME="YOUR_DOCKERHUB_USERNAME"

# Tag the image
docker tag pyinn:latest $DOCKERHUB_USERNAME/pyinn:latest

# Push to Docker Hub
docker push $DOCKERHUB_USERNAME/pyinn:latest
```

Once pushed, users can run your image directly with:
```bash
docker run chanwookpark2024/pyinn:latest
```

## License

This Docker setup follows the same license as the PyINN project (CC BY-NC 4.0). 