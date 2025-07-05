#!/usr/bin/env python3
"""
YOLOv11 training script for unified face and license plate detection
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import argparse
from datetime import datetime

class YOLOTrainer:
    def __init__(self, 
                 model_size: str = "yolov11l.pt",
                 data_config: str = "data.yaml",
                 epochs: int = 100,
                 batch_size: int = 16,
                 img_size: int = 640,
                 device: str = "auto"):
        
        self.model_size = model_size
        self.data_config = data_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.project_name = "pyro-blur"
        self.run_name = f"yolov11l_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Setup device
        self.setup_device()
    
    def setup_device(self):
        """Setup training device"""
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                print("⚠️  Using CPU (GPU not available)")
        
        print(f"Device: {self.device}")
    
    def validate_data_config(self):
        """Validate data configuration"""
        if not Path(self.data_config).exists():
            raise FileNotFoundError(f"Data config file not found: {self.data_config}")
        
        with open(self.data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in data config: {field}")
        
        # Check if data directories exist
        data_path = Path(config['path'])
        for split in ['train', 'val']:
            split_path = data_path / config[split]
            if not split_path.exists():
                raise FileNotFoundError(f"Data directory not found: {split_path}")
        
        print("✅ Data configuration validated")
        return config
    
    def train(self):
        """Train YOLOv11 model"""
        print("🎯 Starting YOLOv11 training...")
        
        # Validate data configuration
        data_config = self.validate_data_config()
        
        # Load model
        print(f"📥 Loading model: {self.model_size}")
        model = YOLO(self.model_size)
        
        # Training parameters
        train_params = {
            'data': self.data_config,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'device': self.device,
            'project': self.project_name,
            'name': self.run_name,
            'save': True,
            'save_period': 10,  # Save every 10 epochs
            'patience': 20,  # Early stopping patience
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'verbose': True
        }
        
        print(f"🏋️  Training parameters:")
        for key, value in train_params.items():
            print(f"  {key}: {value}")
        
        # Start training
        try:
            results = model.train(**train_params)
            
            # Save final model
            final_model_path = self.models_dir / f"{self.run_name}_final.pt"
            model.save(final_model_path)
            
            print(f"✅ Training completed!")
            print(f"📁 Model saved: {final_model_path}")
            print(f"📊 Results: {results}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def export_model(self, model_path: str, formats: list = ['onnx', 'torchscript']):
        """Export trained model to different formats"""
        print(f"📤 Exporting model: {model_path}")
        
        model = YOLO(model_path)
        
        for format_type in formats:
            try:
                exported_path = model.export(format=format_type)
                print(f"✅ Exported to {format_type}: {exported_path}")
            except Exception as e:
                print(f"❌ Failed to export to {format_type}: {e}")
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        print(f"🔄 Resuming training from: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = YOLO(checkpoint_path)
        
        # Resume training with same parameters
        results = model.train(
            data=self.data_config,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            device=self.device,
            project=self.project_name,
            name=f"{self.run_name}_resumed",
            resume=True
        )
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train YOLOv11 for face and license plate detection")
    parser.add_argument("--model", default="yolov11l.pt", help="Model size (yolov11n/s/m/l/x.pt)")
    parser.add_argument("--data", default="data.yaml", help="Data configuration file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    parser.add_argument("--export", help="Export model to different formats")
    parser.add_argument("--export-formats", nargs="+", default=["onnx", "torchscript"], 
                       help="Export formats")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = YOLOTrainer(
        model_size=args.model,
        data_config=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device
    )
    
    # Resume training if specified
    if args.resume:
        trainer.resume_training(args.resume)
    elif args.export:
        trainer.export_model(args.export, args.export_formats)
    else:
        # Start training
        trainer.train()


if __name__ == "__main__":
    main()