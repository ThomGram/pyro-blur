#!/usr/bin/env python3
"""
Benchmark and evaluation script for YOLOv11 face and license plate detection
"""

import os
import sys
import json
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Tuple
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

class ModelBenchmark:
    def __init__(self, model_path: str, data_config: str = "data.yaml"):
        self.model_path = model_path
        self.data_config = data_config
        self.model = None
        self.results = {
            'model_info': {},
            'performance_metrics': {},
            'inference_times': [],
            'detection_results': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Create results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.load_model()
    
    def load_model(self):
        """Load YOLOv11 model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"📥 Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Store model info
        self.results['model_info'] = {
            'model_path': str(self.model_path),
            'model_size': self.model_path,
            'parameters': self._count_parameters(),
            'device': next(self.model.model.parameters()).device.type
        }
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Parameters: {self.results['model_info']['parameters']:,}")
        print(f"🖥️  Device: {self.results['model_info']['device']}")
    
    def _count_parameters(self) -> int:
        """Count model parameters"""
        try:
            return sum(p.numel() for p in self.model.model.parameters())
        except:
            return 0
    
    def validate_model(self):
        """Validate model on test dataset"""
        print("🧪 Validating model on test dataset...")
        
        # Run validation
        validation_results = self.model.val(data=self.data_config, split='test')
        
        # Extract metrics
        metrics = {
            'mAP50': float(validation_results.box.map50),
            'mAP50-95': float(validation_results.box.map),
            'precision': float(validation_results.box.mp),
            'recall': float(validation_results.box.mr),
            'f1_score': 2 * (float(validation_results.box.mp) * float(validation_results.box.mr)) / 
                      (float(validation_results.box.mp) + float(validation_results.box.mr) + 1e-8)
        }
        
        # Per-class metrics
        class_metrics = {}
        if hasattr(validation_results.box, 'maps'):
            class_names = ['face', 'license_plate']
            for i, class_name in enumerate(class_names):
                if i < len(validation_results.box.maps):
                    class_metrics[class_name] = {
                        'mAP50-95': float(validation_results.box.maps[i])
                    }
        
        self.results['performance_metrics'] = {
            'overall': metrics,
            'per_class': class_metrics
        }
        
        print(f"📊 Validation Results:")
        print(f"  mAP50: {metrics['mAP50']:.4f}")
        print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def benchmark_inference_speed(self, test_images_dir: str = None, num_samples: int = 100):
        """Benchmark inference speed"""
        print(f"⚡ Benchmarking inference speed ({num_samples} samples)...")
        
        if test_images_dir is None:
            test_images_dir = Path("data/test/images")
        else:
            test_images_dir = Path(test_images_dir)
        
        if not test_images_dir.exists():
            print("⚠️  Test images directory not found. Creating synthetic test images...")
            self._create_synthetic_test_images(test_images_dir, num_samples)
        
        # Get test images
        image_files = list(test_images_dir.glob("*.jpg"))[:num_samples]
        
        if not image_files:
            print("❌ No test images found")
            return
        
        inference_times = []
        
        # Warmup
        print("🔥 Warming up model...")
        for _ in range(5):
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_img, verbose=False)
        
        # Benchmark
        print("🏁 Running benchmark...")
        for img_path in tqdm(image_files, desc="Processing images"):
            start_time = time.time()
            
            # Run inference
            results = self.model.predict(str(img_path), verbose=False)
            
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Store detection results
            if results:
                detections = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            detections.append({
                                'class': int(box.cls.item()),
                                'confidence': float(box.conf.item()),
                                'bbox': box.xyxy.cpu().numpy().tolist()[0]
                            })
                
                self.results['detection_results'].append({
                    'image': str(img_path),
                    'detections': detections,
                    'inference_time': inference_time
                })
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        fps = 1.0 / avg_time
        
        speed_metrics = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'fps': fps,
            'total_samples': len(inference_times)
        }
        
        self.results['inference_times'] = inference_times
        self.results['speed_metrics'] = speed_metrics
        
        print(f"📊 Inference Speed Results:")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Standard deviation: {std_time:.4f}s")
        print(f"  FPS: {fps:.2f}")
        print(f"  Min time: {np.min(inference_times):.4f}s")
        print(f"  Max time: {np.max(inference_times):.4f}s")
        
        return speed_metrics
    
    def _create_synthetic_test_images(self, test_dir: Path, num_samples: int):
        """Create synthetic test images"""
        test_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            # Create random image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Add some random shapes to make it more realistic
            num_objects = np.random.randint(1, 4)
            for _ in range(num_objects):
                x = np.random.randint(50, 590)
                y = np.random.randint(50, 590)
                w = np.random.randint(50, 150)
                h = np.random.randint(50, 150)
                
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            
            # Save image
            img_path = test_dir / f"test_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)
    
    def analyze_detections(self):
        """Analyze detection results"""
        print("📊 Analyzing detection results...")
        
        if not self.results['detection_results']:
            print("⚠️  No detection results to analyze")
            return
        
        # Count detections by class
        class_counts = {'face': 0, 'license_plate': 0}
        confidence_scores = {'face': [], 'license_plate': []}
        
        for result in self.results['detection_results']:
            for detection in result['detections']:
                class_id = detection['class']
                confidence = detection['confidence']
                
                if class_id == 0:  # face
                    class_counts['face'] += 1
                    confidence_scores['face'].append(confidence)
                elif class_id == 1:  # license_plate
                    class_counts['license_plate'] += 1
                    confidence_scores['license_plate'].append(confidence)
        
        # Calculate statistics
        analysis = {
            'total_detections': sum(class_counts.values()),
            'detections_per_class': class_counts,
            'avg_confidence': {
                'face': np.mean(confidence_scores['face']) if confidence_scores['face'] else 0,
                'license_plate': np.mean(confidence_scores['license_plate']) if confidence_scores['license_plate'] else 0
            },
            'images_with_detections': len([r for r in self.results['detection_results'] if r['detections']])
        }
        
        self.results['detection_analysis'] = analysis
        
        print(f"📈 Detection Analysis:")
        print(f"  Total detections: {analysis['total_detections']}")
        print(f"  Face detections: {analysis['detections_per_class']['face']}")
        print(f"  License plate detections: {analysis['detections_per_class']['license_plate']}")
        print(f"  Images with detections: {analysis['images_with_detections']}")
        
        return analysis
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("📝 Generating benchmark report...")
        
        # Create visualization
        self._create_visualizations()
        
        # Save results to JSON
        results_file = self.results_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate markdown report
        report_file = self.results_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self._generate_markdown_report(report_file)
        
        print(f"📊 Results saved to: {results_file}")
        print(f"📄 Report saved to: {report_file}")
        
        return results_file, report_file
    
    def _create_visualizations(self):
        """Create visualization plots"""
        # Inference time distribution
        if self.results['inference_times']:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(self.results['inference_times'], bins=30, alpha=0.7)
            plt.xlabel('Inference Time (s)')
            plt.ylabel('Frequency')
            plt.title('Inference Time Distribution')
            
            plt.subplot(2, 2, 2)
            plt.plot(self.results['inference_times'])
            plt.xlabel('Sample')
            plt.ylabel('Inference Time (s)')
            plt.title('Inference Time per Sample')
            
            # Performance metrics
            if 'performance_metrics' in self.results:
                plt.subplot(2, 2, 3)
                metrics = self.results['performance_metrics']['overall']
                names = list(metrics.keys())
                values = list(metrics.values())
                plt.bar(names, values)
                plt.title('Performance Metrics')
                plt.xticks(rotation=45)
            
            # Detection counts
            if 'detection_analysis' in self.results:
                plt.subplot(2, 2, 4)
                counts = self.results['detection_analysis']['detections_per_class']
                plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
                plt.title('Detection Distribution')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'benchmark_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_markdown_report(self, report_file: Path):
        """Generate markdown report"""
        with open(report_file, 'w') as f:
            f.write("# YOLOv11 Benchmark Report\n\n")
            f.write(f"Generated: {self.results['timestamp']}\n\n")
            
            # Model info
            f.write("## Model Information\n\n")
            model_info = self.results['model_info']
            f.write(f"- **Model Path**: {model_info['model_path']}\n")
            f.write(f"- **Parameters**: {model_info['parameters']:,}\n")
            f.write(f"- **Device**: {model_info['device']}\n\n")
            
            # Performance metrics
            if 'performance_metrics' in self.results:
                f.write("## Performance Metrics\n\n")
                metrics = self.results['performance_metrics']['overall']
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for name, value in metrics.items():
                    f.write(f"| {name} | {value:.4f} |\n")
                f.write("\n")
            
            # Speed metrics
            if 'speed_metrics' in self.results:
                f.write("## Speed Metrics\n\n")
                speed = self.results['speed_metrics']
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Average Inference Time | {speed['avg_inference_time']:.4f}s |\n")
                f.write(f"| FPS | {speed['fps']:.2f} |\n")
                f.write(f"| Min Time | {speed['min_inference_time']:.4f}s |\n")
                f.write(f"| Max Time | {speed['max_inference_time']:.4f}s |\n")
                f.write("\n")
            
            # Detection analysis
            if 'detection_analysis' in self.results:
                f.write("## Detection Analysis\n\n")
                analysis = self.results['detection_analysis']
                f.write(f"- **Total Detections**: {analysis['total_detections']}\n")
                f.write(f"- **Face Detections**: {analysis['detections_per_class']['face']}\n")
                f.write(f"- **License Plate Detections**: {analysis['detections_per_class']['license_plate']}\n")
                f.write(f"- **Images with Detections**: {analysis['images_with_detections']}\n\n")
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("🚀 Running full benchmark suite...")
        
        # 1. Validate model
        self.validate_model()
        
        # 2. Benchmark inference speed
        self.benchmark_inference_speed()
        
        # 3. Analyze detections
        self.analyze_detections()
        
        # 4. Generate report
        self.generate_report()
        
        print("✅ Benchmark completed successfully!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark YOLOv11 model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", default="data.yaml", help="Data configuration file")
    parser.add_argument("--test-images", help="Directory containing test images")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for speed test")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--speed-only", action="store_true", help="Only run speed benchmark")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = ModelBenchmark(args.model, args.data)
    
    if args.validate_only:
        benchmark.validate_model()
    elif args.speed_only:
        benchmark.benchmark_inference_speed(args.test_images, args.num_samples)
    else:
        benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()