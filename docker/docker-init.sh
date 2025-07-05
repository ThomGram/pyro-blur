#!/bin/bash

# Docker initialization script for PyroBlur

set -e

echo "🐳 Initializing PyroBlur Docker environment..."

# Check if NVIDIA Docker is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi
else
    echo "⚠️  No NVIDIA GPU detected, using CPU mode"
fi

# Function to download dataset
download_dataset() {
    echo "📥 Downloading dataset..."
    python scripts/dataset_downloader.py --max-samples 1000
}

# Function to train model
train_model() {
    echo "🏋️  Training YOLOv11 model..."
    python scripts/train_yolo11.py --epochs 50 --batch 16 --img-size 640
}

# Function to benchmark model
benchmark_model() {
    echo "📊 Benchmarking model..."
    if [ -f "models/yolov11l_*.pt" ]; then
        MODEL_PATH=$(ls models/yolov11l_*.pt | head -1)
        python scripts/benchmark_models.py --model "$MODEL_PATH"
    else
        echo "❌ No trained model found"
        return 1
    fi
}

# Function to run inference
run_inference() {
    echo "🔍 Running inference..."
    if [ -f "models/yolov11l_*.pt" ]; then
        MODEL_PATH=$(ls models/yolov11l_*.pt | head -1)
        python -c "
from ultralytics import YOLO
model = YOLO('$MODEL_PATH')
results = model.predict('data/test/images', save=True)
print('✅ Inference completed, results saved to runs/predict/')
"
    else
        echo "❌ No trained model found"
        return 1
    fi
}

# Parse command line arguments
case "$1" in
    download)
        download_dataset
        ;;
    train)
        train_model
        ;;
    benchmark)
        benchmark_model
        ;;
    inference)
        run_inference
        ;;
    full)
        echo "🚀 Running full pipeline..."
        download_dataset
        train_model
        benchmark_model
        run_inference
        ;;
    bash)
        echo "🐚 Starting interactive bash session..."
        /bin/bash
        ;;
    jupyter)
        echo "📓 Starting Jupyter notebook server..."
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;
    *)
        echo "Usage: $0 {download|train|benchmark|inference|full|bash|jupyter}"
        echo ""
        echo "Commands:"
        echo "  download   - Download and prepare dataset"
        echo "  train      - Train YOLOv11 model"
        echo "  benchmark  - Benchmark trained model"
        echo "  inference  - Run inference on test images"
        echo "  full       - Run complete pipeline"
        echo "  bash       - Start interactive bash session"
        echo "  jupyter    - Start Jupyter notebook server"
        exit 1
        ;;
esac