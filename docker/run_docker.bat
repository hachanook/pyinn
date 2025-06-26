@echo off

REM Build the Docker image
echo Building Docker image...
docker build -f docker/Dockerfile -t pyinn .

REM Run the container (CPU-only, compatible with all systems)
echo Running pyinn container...
docker run ^
    -v %cd%/data:/app/data ^
    -v %cd%/plots:/app/plots ^
    -v %cd%/config:/app/config ^
    -p 8888:8888 ^
    pyinn

pause 