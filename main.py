#!/usr/bin/env python3
"""
Main entry point for PyroBlur - YOLOv11 Face and License Plate Detection
"""

import argparse
import sys
from pathlib import Path
from scripts.dataset_downloader import DatasetDownloader
from scripts.train_yolo11 import YOLOTrainer
from scripts.benchmark_models import ModelBenchmark
from ultralytics import YOLO

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="PyroBlur - YOLOv11 Face and License Plate Detection")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download dataset')
    download_parser.add_argument('--max-samples', type=int, default=1000, help='Max samples per class')
    download_parser.add_argument('--roboflow-key', help='Roboflow API key (optional)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train YOLOv11 model')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--device', default='auto', help='Device (auto/cpu/cuda)')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark trained model')
    benchmark_parser.add_argument('--model', required=True, help='Path to trained model')
    benchmark_parser.add_argument('--num-samples', type=int, default=100, help='Number of samples for speed test')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--model', required=True, help='Path to trained model')
    inference_parser.add_argument('--source', required=True, help='Input source (image/video/directory)')
    inference_parser.add_argument('--output', default='results', help='Output directory')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--max-samples', type=int, default=1000, help='Max samples per class')
    full_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    full_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'download':
            print("📥 Downloading dataset...")
            downloader = DatasetDownloader(max_samples=args.max_samples)
            downloader.download_all(args.roboflow_key)
            
        elif args.command == 'train':
            print("🏋️  Training YOLOv11 model...")
            trainer = YOLOTrainer(
                epochs=args.epochs,
                batch_size=args.batch,
                device=args.device
            )
            trainer.train()
            
        elif args.command == 'benchmark':
            print("📊 Benchmarking model...")
            benchmark = ModelBenchmark(args.model)
            benchmark.run_full_benchmark()
            
        elif args.command == 'inference':
            print("🔍 Running inference...")
            model = YOLO(args.model)
            results = model.predict(
                source=args.source,
                save=True,
                project=args.output,
                name='inference',
                conf=0.5,
                imgsz=640
            )
            print(f"✅ Inference completed, results saved to {args.output}/inference/")
            
        elif args.command == 'full':
            print("🚀 Running full pipeline...")
            
            # Step 1: Download dataset
            print("Step 1/4: Downloading dataset...")
            downloader = DatasetDownloader(max_samples=args.max_samples)
            downloader.download_all()
            
            # Step 2: Train model
            print("Step 2/4: Training model...")
            trainer = YOLOTrainer(
                epochs=args.epochs,
                batch_size=args.batch
            )
            trainer.train()
            
            # Step 3: Find trained model
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pt"))
                if model_files:
                    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                    
                    # Step 4: Benchmark model
                    print("Step 3/4: Benchmarking model...")
                    benchmark = ModelBenchmark(str(latest_model))
                    benchmark.run_full_benchmark()
                    
                    # Step 5: Run sample inference
                    print("Step 4/4: Running sample inference...")
                    model = YOLO(str(latest_model))
                    results = model.predict(
                        source="data/test/images",
                        save=True,
                        project="results",
                        name="final_inference",
                        conf=0.5,
                        imgsz=640
                    )
                    
                    print("✅ Full pipeline completed successfully!")
                else:
                    print("❌ No trained model found")
            else:
                print("❌ Models directory not found")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()