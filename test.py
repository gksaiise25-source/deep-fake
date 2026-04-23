"""
Test Script for Deepfake Detection System

This script tests all components of the system
"""

import os
import sys
import numpy as np
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import cv2
        logger.info("✓ OpenCV imported successfully")
    except ImportError as e:
        logger.error(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("✓ NumPy imported successfully")
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        logger.info(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        logger.error(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        from data_preprocessing import DatasetLoader, FaceDetector
        logger.info("✓ data_preprocessing module imported successfully")
    except ImportError as e:
        logger.error(f"✗ data_preprocessing import failed: {e}")
        return False
    
    try:
        from model import DeepfakeDetectionModel
        logger.info("✓ model module imported successfully")
    except ImportError as e:
        logger.error(f"✗ model import failed: {e}")
        return False
    
    try:
        from train import ModelTrainer
        logger.info("✓ train module imported successfully")
    except ImportError as e:
        logger.error(f"✗ train import failed: {e}")
        return False
    
    try:
        from predict import DeepfakePredictor
        logger.info("✓ predict module imported successfully")
    except ImportError as e:
        logger.error(f"✗ predict import failed: {e}")
        return False
    
    return True


def test_data_preprocessing():
    """Test data preprocessing module"""
    logger.info("\nTesting data preprocessing...")
    
    from data_preprocessing import FaceDetector, ImagePreprocessor, VideoProcessor
    
    # Test FaceDetector
    try:
        detector = FaceDetector(method='opencv')
        logger.info("✓ FaceDetector initialized")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(test_image)
        logger.info(f"✓ Face detection works (found {len(faces)} faces)")
    except Exception as e:
        logger.error(f"✗ FaceDetector test failed: {e}")
        return False
    
    # Test ImagePreprocessor
    try:
        processor = ImagePreprocessor(target_size=(256, 256))
        logger.info("✓ ImagePreprocessor initialized")
        
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        processed = processor.preprocess(test_image)
        assert processed.shape == (256, 256, 3)
        assert processed.min() >= 0 and processed.max() <= 1
        logger.info("✓ Image preprocessing works")
    except Exception as e:
        logger.error(f"✗ ImagePreprocessor test failed: {e}")
        return False
    
    return True


def test_model():
    """Test model module"""
    logger.info("\nTesting model module...")
    
    from model import DeepfakeDetectionModel
    
    try:
        # Test model initialization
        model = DeepfakeDetectionModel(model_type='custom')
        logger.info("✓ Custom CNN model initialized")
        
        # Test compilation
        model.compile()
        logger.info("✓ Model compiled successfully")
        
        # Test model summary
        model.summary()
        logger.info("✓ Model summary generated")
        
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False
    
    return True


def test_training_basic():
    """Test training with minimal data"""
    logger.info("\nTesting training module...")
    
    from train import ModelTrainer
    import numpy as np
    
    try:
        # Create minimal test data
        logger.info("Creating test data...")
        X_test = np.random.rand(10, 256, 256, 3).astype(np.float32)
        y_test = np.array([0, 1] * 5)
        
        # Test trainer
        trainer = ModelTrainer(model_type='custom')
        logger.info("✓ ModelTrainer initialized")
        
        # Prepare data
        data = trainer.prepare_data(X_test, y_test, test_size=0.2, val_size=0.25)
        logger.info("✓ Data prepared successfully")
        
        # Check shapes
        assert data['X_train'].shape[0] > 0
        assert data['X_val'].shape[0] > 0
        assert data['X_test'].shape[0] > 0
        logger.info(f"✓ Data split: Train={len(data['X_train'])}, Val={len(data['X_val'])}, Test={len(data['X_test'])}")
        
    except Exception as e:
        logger.error(f"✗ Training test failed: {e}")
        return False
    
    return True


def test_directory_structure():
    """Test if required directories exist"""
    logger.info("\nTesting directory structure...")
    
    required_dirs = ['data', 'models', 'outputs']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            logger.info(f"✓ {dir_name}/ directory exists")
        else:
            logger.warning(f"✗ {dir_name}/ directory missing (will be created on demand)")
    
    return True


def test_sample_prediction():
    """Test prediction on sample data"""
    logger.info("\nTesting prediction capability...")
    
    try:
        from predict import DeepfakePredictor
        from data_preprocessing import ImagePreprocessor
        import tempfile
        import cv2
        
        # Create a temporary test image
        logger.info("Creating test image...")
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            tmp_path = tmp.name
        
        try:
            # Test will fail if model doesn't exist, which is expected
            logger.info("Note: Prediction test requires a trained model")
            logger.info("To complete this test, train a model first: python train.py --create-sample")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Prediction test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Deepfake Detection System - Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Architecture", test_model),
        ("Training Setup", test_training_basic),
        ("Directory Structure", test_directory_structure),
        ("Prediction Setup", test_sample_prediction)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            failed += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    if failed == 0:
        logger.info("\n✅ All tests passed! System is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Create sample data: python train.py --create-sample")
        logger.info("2. Train model: python train.py --create-sample --epochs 20")
        logger.info("3. Launch web app: streamlit run app.py")
        return 0
    else:
        logger.info(f"\n❌ {failed} test(s) failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
