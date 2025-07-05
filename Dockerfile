# YOLOv11 Face and License Plate Detection Docker Image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set non-interactive mode and timezone to avoid tzdata configuration prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data models results

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "scripts/dataset_downloader.py", "--max-samples", "1000"]