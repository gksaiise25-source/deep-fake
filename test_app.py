"""
Demo Testing Script - Batch Prediction & Analysis

Test the deepfake detection model on images.
Supports:
- Single image prediction
- Batch prediction from directory
- Performance visualization
- Confidence threshold analysis

Run with:
    python test_app.py --image path/to/image.jpg
    python test_app.py --directory path/to/images/
    python test_app.py --analyze  (analyze all predictions)
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


class DeepfakeTester:
    """Test deepfake detection model"""
    
    def __init__(self, model_type='sklearn'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Load the model"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if self.model_type == 'pytorch':
            try:
                import torch
                from model_deep_learning import DeepfakeDetectionModel
                
                self.model = DeepfakeDetectionModel()
                model_path = os.path.join(base_dir, 'models', 'deepfake_resnet18.pth')
                
                if os.path.exists(model_path):
                    self.model.build_model()
                    self.model.load_model(model_path)
                else:
                    logger.error(f"PyTorch model not found at {model_path}")
                    logger.info("Train with: python train_enhanced.py --model pytorch")
                    sys.exit(1)
            except ImportError:
                logger.error("PyTorch not installed")
                logger.info("Install with: pip install torch torchvision")
                sys.exit(1)
        else:
            # Load sklearn model
            from model_advanced import AdvancedDeepfakeDetectionModel
            
            self.model = AdvancedDeepfakeDetectionModel()
            model_path = os.path.join(base_dir, 'models', 'deepfake_advanced_ra.pkl')
            
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                logger.info("Train with: python train_advanced.py")
                sys.exit(1)
            
            self.model.load(model_path)
            self.preprocessor = self.model.preprocessor
    
    def predict_image(self, image_path):
        """
        Predict if image is fake
        
        Returns:
            result dict with prediction, confidence, etc.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        prediction, probabilities = self.model.predict(image)
        
        if prediction is None:
            return None
        
        is_fake = prediction == 1
        confidence = max(probabilities) * 100
        
        return {
            'path': image_path,
            'filename': os.path.basename(image_path),
            'is_fake': is_fake,
            'prediction': 'FAKE' if is_fake else 'REAL',
            'confidence': confidence,
            'real_prob': probabilities[0] * 100,
            'fake_prob': probabilities[1] * 100
        }
    
    def test_single_image(self, image_path):
        """Test single image"""
        result = self.predict_image(image_path)
        
        if result is None:
            return
        
        print("\n" + "="*60)
        print("🔍 DEEPFAKE DETECTION RESULT")
        print("="*60)
        print(f"📁 File: {result['filename']}")
        print(f"\n{'🚨' if result['is_fake'] else '✅'} Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2f}%")
        print(f"   Real:     {result['real_prob']:.2f}%")
        print(f"   Fake:     {result['fake_prob']:.2f}%")
        print("="*60 + "\n")
        
        # Display image
        try:
            img = Image.open(image_path)
            img.show()
        except:
            pass
    
    def test_directory(self, directory, output_csv=None):
        """Test all images in directory"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(directory).glob(f'**/{ext}'))
            image_files.extend(Path(directory).glob(f'**/{ext.upper()}'))
        
        results = []
        
        print(f"\n🔍 Testing {len(image_files)} images...\n")
        
        for image_path in image_files:
            result = self.predict_image(str(image_path))
            if result:
                results.append(result)
                status = '🚨 FAKE' if result['is_fake'] else '✅ REAL'
                print(f"{status} | {result['filename']:30s} | {result['confidence']:6.2f}%")
        
        # Summary
        if results:
            fake_count = sum(1 for r in results if r['is_fake'])
            real_count = len(results) - fake_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print("\n" + "="*60)
            print("📊 SUMMARY")
            print("="*60)
            print(f"✅ Real:  {real_count}/{len(results)}")
            print(f"🚨 Fake:  {fake_count}/{len(results)}")
            print(f"📈 Avg Confidence: {avg_confidence:.2f}%")
            print("="*60 + "\n")
            
            # Save results
            if output_csv:
                import csv
                with open(output_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                print(f"💾 Results saved to: {output_csv}\n")
        
        return results
    
    def analyze_predictions(self, results):
        """Analyze and visualize prediction results"""
        if not results:
            return
        
        # Confidence distribution
        confidences = [r['confidence'] for r in results]
        predictions = [r['prediction'] for r in results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(confidences, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Confidence (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Pie chart
        unique, counts = np.unique(predictions, return_counts=True)
        colors = ['#FF6B6B' if u == 'FAKE' else '#4ECDC4' for u in unique]
        axes[1].pie(counts, labels=unique, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1].set_title('Prediction Distribution')
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=150)
        print("📊 Analysis saved to: prediction_analysis.png\n")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Tester')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--analyze', action='store_true', help='Analyze predictions')
    parser.add_argument('--model', choices=['sklearn', 'pytorch'], default='sklearn',
                       help='Model to use (default: sklearn)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = DeepfakeTester(model_type=args.model)
    
    if args.image:
        # Single image
        tester.test_single_image(args.image)
    
    elif args.directory:
        # Directory
        results = tester.test_directory(args.directory, output_csv=args.output)
        
        if args.analyze:
            tester.analyze_predictions(results)
    
    else:
        # Default: test data directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(base_dir, 'data')
        
        if os.path.exists(test_dir):
            results = tester.test_directory(test_dir)
            
            if args.analyze:
                tester.analyze_predictions(results)
        else:
            logger.error("No image path provided. Use --image or --directory")
            parser.print_help()


if __name__ == "__main__":
    main()
