# Use official Miniconda base image for CPU-only support (compatible with all systems)
FROM continuumio/miniconda3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISABLE_VTUNE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy environment file first for better caching
COPY docker/environment_docker.yaml .

# Create conda environment
RUN conda env create -f environment_docker.yaml

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "pyinn-env", "/bin/bash", "-c"]

# Install PyTorch CPU version via pip (more reliable than conda)
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install JAX with CPU support (compatible with all systems)
RUN pip install -U jax

# Install Optax (optimization library for JAX)
RUN pip install optax

# Install Jupyter for notebook support
RUN pip install jupyter jupyterlab

# Copy the entire project
COPY . .

# Install the local package in development mode (this ensures we use the local code)
RUN pip install -e .

# Set the default command to run the main script
CMD ["conda", "run", "-n", "pyinn-env", "python", "./pyinn/main.py"] 