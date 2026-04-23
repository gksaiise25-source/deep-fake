"""
Advanced Deepfake Detection Model using scikit-learn

Uses advanced feature extraction for better accuracy.
"""

import os
import numpy as np
import cv2
import joblib
import logging
from advanced_features import AdvancedFeatureExtractor
from data_preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class AdvancedDeepfakeDetectionModel:
    """Advanced model for deepfake detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_extractor = AdvancedFeatureExtractor()
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))
    
    def load(self, model_path, scaler_path=None):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
        
        # If scaler path not provided, infer it from model directory
        if scaler_path is None:
            model_dir = os.path.dirname(model_path)
            scaler_path = os.path.join(model_dir, 'scaler_advanced.pkl')
        
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            logger.warning(f"Scaler not found at {scaler_path}, using identity scaling")
            self.scaler = None
        
        return True
    
    def predict(self, image_array):
        """
        Predict if image is real or fake
        
        Args:
            image_array: Image as numpy array (0-1 range) or (0-255 range)
        
        Returns:
            prediction: 0 = REAL, 1 = FAKE
            confidence: Confidence score (0-1)
        """
        if self.model is None:
            logger.error("Model not loaded!")
            return None, 0.0
        
        try:
            # Ensure image is in 0-1 range
            if image_array.max() > 1:
                image_array = image_array / 255.0
            
            # Preprocess
            processed = self.preprocessor.preprocess(image_array)
            
            # Extract features
            features = self.feature_extractor.extract_features(processed)
            features = features.reshape(1, -1)
            
            # Scale if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features)[0]
            confidence = self.model.predict_proba(features)[0]
            
            return prediction, confidence
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, None
    
    def predict_directory(self, image_dir):
        """Batch predict on directory of images"""
        results = []
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                filepath = os.path.join(image_dir, filename)
                image = cv2.imread(filepath)
                
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
                    pred, conf = self.predict(image)
                    
                    results.append({
                        'filename': filename,
                        'prediction': 'FAKE' if pred == 1 else 'REAL',
                        'real_confidence': float(conf[0]),
                        'fake_confidence': float(conf[1])
                    })
        
        return results
