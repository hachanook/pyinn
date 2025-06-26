@echo off

REM Script to push PyINN Docker image to Docker Hub
REM Replace YOUR_DOCKERHUB_USERNAME with your actual Docker Hub username

REM Set your Docker Hub username here
set DOCKERHUB_USERNAME=chanwookpark2024
set IMAGE_NAME=pyinn
set VERSION=latest

echo === Pushing PyINN to Docker Hub ===
echo Docker Hub username: %DOCKERHUB_USERNAME%
echo Image name: %IMAGE_NAME%
echo Version: %VERSION%
echo.

REM Check if the local image exists
docker image inspect %IMAGE_NAME%:%VERSION% >nul 2>&1
if errorlevel 1 (
    echo ❌ Local image '%IMAGE_NAME%:%VERSION%' not found!
    echo Please build the image first with: docker build -f docker/Dockerfile -t %IMAGE_NAME% .
    pause
    exit /b 1
)

REM Tag the image for Docker Hub
echo 🏷️  Tagging image...
docker tag %IMAGE_NAME%:%VERSION% %DOCKERHUB_USERNAME%/%IMAGE_NAME%:%VERSION%

REM Push to Docker Hub
echo 🚀 Pushing to Docker Hub...
docker push %DOCKERHUB_USERNAME%/%IMAGE_NAME%:%VERSION%

if errorlevel 0 (
    echo ✅ Successfully pushed to Docker Hub!
    echo 🔗 Your image is now available at: https://hub.docker.com/r/%DOCKERHUB_USERNAME%/%IMAGE_NAME%
    echo.
    echo Others can now run your image with:
    echo docker run %DOCKERHUB_USERNAME%/%IMAGE_NAME%:%VERSION%
) else (
    echo ❌ Failed to push to Docker Hub
)

pause 