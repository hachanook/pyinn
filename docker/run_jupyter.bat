@echo off

REM Script to run Jupyter notebooks in PyINN Docker container

echo 🚀 Starting PyINN Jupyter Environment...
echo 📚 Tutorials will be available in the /tutorials directory
echo 🌐 Jupyter will be accessible at http://localhost:8888
echo.

REM Build the Docker image if it doesn't exist
docker image inspect pyinn:latest >nul 2>&1
if errorlevel 1 (
    echo 🔨 Building Docker image...
    docker build -f docker/Dockerfile -t pyinn .
    echo.
)

REM Run Jupyter in the container
echo 🎯 Starting Jupyter Lab...
docker run -it --rm ^
    -p 8888:8888 ^
    -v %cd%/tutorials:/app/tutorials ^
    -v %cd%/data:/app/data ^
    -v %cd%/plots:/app/plots ^
    -v %cd%/config:/app/config ^
    pyinn ^
    conda run -n pyinn-env jupyter lab ^
        --ip=0.0.0.0 ^
        --port=8888 ^
        --no-browser ^
        --allow-root ^
        --notebook-dir=/app

pause 