#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -f docker/Dockerfile -t pyinn .

# Run the container (CPU-only, compatible with all systems)
echo "Running pyinn container..."
docker run \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/plots:/app/plots \
    -v $(pwd)/config:/app/config \
    -p 8888:8888 \
    pyinn 