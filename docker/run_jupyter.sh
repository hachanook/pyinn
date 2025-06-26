#!/bin/bash

# Script to run Jupyter notebooks in PyINN Docker container

echo "🚀 Starting PyINN Jupyter Environment..."
echo "📚 Tutorials will be available in the /tutorials directory"
echo "🌐 Jupyter will be accessible at http://localhost:8888"
echo

# Build the Docker image if it doesn't exist
if ! docker image inspect pyinn:latest >/dev/null 2>&1; then
    echo "🔨 Building Docker image..."
    docker build -f docker/Dockerfile -t pyinn .
    echo
fi

# Run Jupyter in the container
echo "🎯 Starting Jupyter Lab..."
docker run -it --rm \
    -p 8888:8888 \
    -v $(pwd)/tutorials:/app/tutorials \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/plots:/app/plots \
    -v $(pwd)/config:/app/config \
    pyinn \
    conda run -n pyinn-env jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --notebook-dir=/app 