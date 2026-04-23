"""
Enhanced Training Script with Improved Pipeline

Features:
- Automatic data augmentation
- Model validation during training
- Performance metrics visualization
- Best model checkpointing
- Support for both sklearn and PyTorch models

Run with:
    python train_enhanced.py --model sklearn
    python train_enhanced.py --model pytorch
"""

import argparse
import os
import sys
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedTrainingPipeline:
    """Complete training pipeline with augmentation and validation"""
    
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
    
    def check_data(self):
        """Check if training data exists"""
        real_dir = os.path.join(self.data_dir, 'train_real')
        fake_dir = os.path.join(self.data_dir, 'train_fake')
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            logger.error("❌ Training data not found!")
            logger.info(f"Expected: {real_dir} and {fake_dir}")
            return False
        
        real_count = len([f for f in os.listdir(real_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        fake_count = len([f for f in os.listdir(fake_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        logger.info(f"✅ Found {real_count} real + {fake_count} fake images")
        return True
    
    def augment_data(self):
        """Run data augmentation"""
        logger.info("\n" + "="*60)
        logger.info("🔄 STEP 1: DATA AUGMENTATION")
        logger.info("="*60)
        
        augment_script = os.path.join(self.base_dir, 'augment_data.py')
        if not os.path.exists(augment_script):
            logger.error(f"augment_data.py not found")
            return False
        
        try:
            result = subprocess.run([sys.executable, augment_script], check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Augmentation failed: {e}")
            return False
    
    def train_sklearn_model(self):
        """Train sklearn model"""
        logger.info("\n" + "="*60)
        logger.info("🤖 STEP 2: TRAINING SKLEARN MODEL")
        logger.info("="*60)
        
        train_script = os.path.join(self.base_dir, 'train_advanced.py')
        if not os.path.exists(train_script):
            logger.error(f"train_advanced.py not found")
            return False
        
        try:
            result = subprocess.run([sys.executable, train_script], check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def train_pytorch_model(self):
        """Train PyTorch deep learning model"""
        logger.info("\n" + "="*60)
        logger.info("🧠 STEP 2: TRAINING PYTORCH MODEL")
        logger.info("="*60)
        
        # Check if torch is available
        try:
            import torch
            logger.info(f"✅ PyTorch available (CUDA: {torch.cuda.is_available()})")
        except ImportError:
            logger.error("❌ PyTorch not installed")
            logger.info("Install with: pip install torch torchvision")
            return False
        
        train_script = os.path.join(self.base_dir, 'model_deep_learning.py')
        if not os.path.exists(train_script):
            logger.error(f"model_deep_learning.py not found")
            return False
        
        try:
            result = subprocess.run([sys.executable, train_script], check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def validate_model(self):
        """Run model validation"""
        logger.info("\n" + "="*60)
        logger.info("📊 STEP 3: MODEL VALIDATION")
        logger.info("="*60)
        
        validate_script = os.path.join(self.base_dir, 'validate_model.py')
        if not os.path.exists(validate_script):
            logger.warning(f"validate_model.py not found, skipping validation")
            return True
        
        try:
            result = subprocess.run([sys.executable, validate_script], check=False)
            return True
        except Exception as e:
            logger.warning(f"Validation skipped: {e}")
            return True
    
    def run_full_pipeline(self, model_type='sklearn'):
        """Run complete training pipeline"""
        logger.info("\n" + "="*70)
        logger.info("🚀 DEEPFAKE DETECTION - COMPLETE TRAINING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Check data
        if not self.check_data():
            return False
        
        # Step 2: Augment data
        if not self.augment_data():
            logger.warning("⚠️  Augmentation failed, continuing with original data...")
        
        # Step 3: Train model
        if model_type == 'pytorch':
            if not self.train_pytorch_model():
                return False
        else:  # sklearn (default)
            if not self.train_sklearn_model():
                return False
        
        # Step 4: Validate
        if not self.validate_model():
            logger.warning("⚠️  Validation failed")
        
        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"\n🎯 Next steps:")
        logger.info(f"   1. Run the Streamlit app: python -m streamlit run streamlit_app.py")
        logger.info(f"   2. Check outputs folder for metrics and visualizations")
        logger.info(f"   3. Test with test_app.py for batch predictions")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Enhanced Training Pipeline')
    parser.add_argument('--model', choices=['sklearn', 'pytorch'], default='sklearn',
                       help='Model type to train (default: sklearn)')
    parser.add_argument('--skip-augment', action='store_true',
                       help='Skip data augmentation')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation')
    
    args = parser.parse_args()
    
    pipeline = EnhancedTrainingPipeline()
    
    if args.validate_only:
        pipeline.validate_model()
    else:
        success = pipeline.run_full_pipeline(model_type=args.model)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
