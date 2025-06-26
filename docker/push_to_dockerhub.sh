#!/bin/bash

# Script to push PyINN Docker image to Docker Hub
# Replace YOUR_DOCKERHUB_USERNAME with your actual Docker Hub username

# Set your Docker Hub username here
DOCKERHUB_USERNAME="chanwookpark2024"
IMAGE_NAME="pyinn"
VERSION="latest"

echo "=== Pushing PyINN to Docker Hub ==="
echo "Docker Hub username: $DOCKERHUB_USERNAME"
echo "Image name: $IMAGE_NAME"
echo "Version: $VERSION"
echo

# Check if the local image exists
if ! docker image inspect $IMAGE_NAME:$VERSION >/dev/null 2>&1; then
    echo "❌ Local image '$IMAGE_NAME:$VERSION' not found!"
    echo "Please build the image first with: docker build -f docker/Dockerfile -t $IMAGE_NAME ."
    exit 1
fi

# Tag the image for Docker Hub
echo "🏷️  Tagging image..."
docker tag $IMAGE_NAME:$VERSION $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION

# Push to Docker Hub
echo "🚀 Pushing to Docker Hub..."
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to Docker Hub!"
    echo "🔗 Your image is now available at: https://hub.docker.com/r/$DOCKERHUB_USERNAME/$IMAGE_NAME"
    echo
    echo "Others can now run your image with:"
    echo "docker run $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION"
else
    echo "❌ Failed to push to Docker Hub"
    exit 1
fi 