"""
Machine Learning Model for Deepfake Detection

Uses scikit-learn for Python 3.14 compatibility
Implements fast feature extraction and classification
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import cv2

logger = logging.getLogger(__name__)


class DeepfakeDetectionModel:
    """
    Deepfake detection model using scikit-learn.
    Works with Python 3.14+
    """
    
    def __init__(self, model_type: str = 'random_forest', input_shape: tuple = (256, 256, 3)):
        """
        Initialize the model
        
        Args:
            model_type (str): Type of model - 'random_forest', 'gradient_boost'
            input_shape (tuple): Input image shape (height, width, channels)
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Model {model_type} initialized with input shape {input_shape}")
    
    def _extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images using histogram and texture analysis
        
        Args:
            images (np.ndarray): Array of images (N, H, W, 3)
            
        Returns:
            Feature vectors (N, n_features)
        """
        n_images = images.shape[0]
        features_list = []
        
        for img in images:
            # Ensure image is in correct format
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Convert to BGR for OpenCV
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            
            # Extract histogram features
            features = []
            
            # Color histogram
            for i in range(3):
                hist = cv2.calcHist([img_bgr], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # Grayscale histogram
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256])
            features.extend(hist_gray.flatten())
            
            # Laplacian variance (detects blur)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(laplacian_var)
            
            # Mean and std of pixel values
            features.append(gray.mean())
            features.append(gray.std())
            
            # Edge detection features
            canny = cv2.Canny(gray, 100, 200)
            features.append(canny.sum() / canny.size)
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model on training data
        
        Args:
            X (np.ndarray): Training images
            y (np.ndarray): Training labels
        """
        logger.info(f"Extracting features from {len(X)} images...")
        X_features = self._extract_features(X)
        
        logger.info(f"Scaling features...")
        X_scaled = self.scaler.fit_transform(X_features)
        
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Model training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X (np.ndarray): Images to predict on
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_features = self._extract_features(X)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions
        
        Args:
            X (np.ndarray): Images to predict on
            
        Returns:
            Probability array (N, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_features = self._extract_features(X)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict_proba(X_scaled)
    
    def get_model(self):
        """Get the sklearn model"""
        return self.model
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Get accuracy score
        
        Args:
            X (np.ndarray): Test images
            y (np.ndarray): Test labels
            
        Returns:
            Accuracy score
        """
        X_features = self._extract_features(X)
        X_scaled = self.scaler.transform(X_features)
        return self.model.score(X_scaled, y)
    
    def save(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save model
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'input_shape': self.input_shape
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath (str): Path to load model from
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        self.input_shape = data['input_shape']
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary"""
        print(f"\n{'='*60}")
        print(f"Model: {self.model_type}")
        print(f"Input Shape: {self.input_shape}")
        print(f"Fitted: {self.is_fitted}")
        if self.model:
            print(f"Model: {self.model}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    model = DeepfakeDetectionModel(model_type='random_forest')
    model.summary()
