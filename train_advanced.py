"""
Advanced Training Script for Deepfake Detection

Train improved model using advanced feature extraction.

Usage:
    python train_advanced.py --create-sample      # Train with synthetic data
    python train_advanced.py --data path/to/data   # Train with custom data
"""

import os
import sys
import numpy as np
import cv2
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import json
from advanced_features import AdvancedFeatureExtractor
from data_preprocessing import ImagePreprocessor

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedDeepfakeDetectionTrainer:
    """Train deepfake detection model with advanced features"""
    
    def __init__(self, model_type='random_forest'):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))
        self.scaler = StandardScaler()
        self.model = None
        self.model_type = model_type
        self.results = {}
        
        # Create models directory if needed
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
    
    def create_sample_data(self, num_real=50, num_fake=50):
        """
        Create sample training data with realistic features
        Real images: photos with natural variations
        Fake images: AI-generated or with synthetic artifacts
        """
        print(f"Creating {num_real} real and {num_fake} fake sample images...")
        
        real_images = []
        fake_images = []
        
        # Create realistic real images
        for i in range(num_real):
            # Natural photograph simulation
            img = np.random.rand(256, 256, 3) * 0.3 + 0.4  # Base intensity
            
            # Add natural variations
            # Add noise patterns
            noise = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)
            
            # Add texture (natural scenes have texture)
            x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
            texture = 0.1 * np.sin(10 * x) * np.cos(10 * y)
            texture = np.stack([texture] * 3, axis=2)
            img = np.clip(img + texture * 0.1, 0, 1)
            
            # Add realistic lighting variations
            lighting = 0.5 + 0.3 * np.sin(np.pi * x) * np.cos(np.pi * y)
            lighting = np.stack([lighting] * 3, axis=2)
            img = img * lighting
            img = np.clip(img, 0, 1)
            
            # Add some complexity with blob-like structures
            for _ in range(2):
                cy, cx = np.random.randint(50, 206, 2)
                radius = np.random.randint(10, 40)
                y, x = np.ogrid[:256, :256]
                mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                img[mask] = np.clip(img[mask] * np.random.uniform(0.8, 1.2), 0, 1)
            
            real_images.append(img)
        
        # Create AI-generated looking images (with characteristic artifacts)
        for i in range(num_fake):
            # Smoother base (typical of AI generation)
            img = np.random.rand(256, 256, 3) * 0.2 + 0.5
            
            # Apply Gaussian blur for synthetic smoothness (AI artifact)
            img = cv2.GaussianBlur((img * 255).astype(np.uint8), (5, 5), 1.5)
            img = img.astype(np.float32) / 255.0
            
            # Add regular grid patterns (common in some GANs)
            grid_freq = np.random.randint(4, 8)
            x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
            grid = 0.1 * np.sin(grid_freq * np.pi * x) * np.sin(grid_freq * np.pi * y)
            grid = np.stack([grid] * 3, axis=2)
            img = np.clip(img + grid, 0, 1)
            
            # Add slight color banding (common in AI-generated content)
            color_bands = np.random.rand(5, 3)
            for j in range(5):
                start_y = j * 51
                end_y = (j + 1) * 51
                img[start_y:end_y] = np.clip(
                    img[start_y:end_y] * color_bands[j],
                    0, 1
                )
            
            # Add smoothed artifacts (block-like patterns)
            for _ in range(2):
                cy, cx = np.random.randint(50, 206, 2)
                size = np.random.randint(20, 60)
                rect_color = np.random.rand(3)
                img[cy:cy+size, cx:cx+size] = np.clip(
                    img[cy:cy+size, cx:cx+size] * 0.7 + rect_color * 0.3,
                    0, 1
                )
            
            # Blur this fake region to make it look synthesized
            img = cv2.GaussianBlur((img * 255).astype(np.uint8), (3, 3), 0.5)
            img = img.astype(np.float32) / 255.0
            
            fake_images.append(img)
        
        return real_images, fake_images
    
    def prepare_data(self, real_images, fake_images):
        """Prepare training data"""
        print("Extracting features...")
        
        X = []
        y = []
        
        # Process real images
        for i, img in enumerate(real_images):
            processed = self.preprocessor.preprocess(img)
            features = self.feature_extractor.extract_features(processed)
            X.append(features)
            y.append(0)  # 0 = REAL
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(real_images)} real images")
        
        # Process fake images
        for i, img in enumerate(fake_images):
            processed = self.preprocessor.preprocess(img)
            features = self.feature_extractor.extract_features(processed)
            X.append(features)
            y.append(1)  # 1 = FAKE
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(fake_images)} fake images")
        
        X = np.array(X)
        y = np.array(y)
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        print("Training complete!")
    
    def evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Evaluate model performance"""
        print("\n=== Model Evaluation ===\n")
        
        self.results = {}
        
        for split_name, X, y in [
            ("Train", X_train, y_train),
            ("Validation", X_val, y_val),
            ("Test", X_test, y_test)
        ]:
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            
            self.results[split_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc)
            }
            
            print(f"{split_name} Set:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC:       {auc:.4f}\n")
    
    def save_model(self, model_name='deepfake_advanced_rf.pkl'):
        """Save the trained model"""
        model_path = f'models/{model_name}'
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = f'models/scaler_advanced.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Save results
        results_path = 'outputs/advanced_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Train advanced deepfake detection model')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data')
    parser.add_argument('--model-type', default='random_forest', choices=['random_forest', 'gradient_boosting'])
    parser.add_argument('--num-real', type=int, default=75, help='Number of real samples')
    parser.add_argument('--num-fake', type=int, default=75, help='Number of fake samples')
    
    args = parser.parse_args()
    
    trainer = AdvancedDeepfakeDetectionTrainer(model_type=args.model_type)
    
    if args.create_sample:
        real_imgs, fake_imgs = trainer.create_sample_data(
            num_real=args.num_real,
            num_fake=args.num_fake
        )
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(real_imgs, fake_imgs)
    
    trainer.train(X_train, y_train)
    trainer.evaluate(X_train, X_val, X_test, y_train, y_val, y_test)
    trainer.save_model(f'deepfake_advanced_{args.model_type[:2]}.pkl')
    
    print("\n✅ Training pipeline complete!")


if __name__ == '__main__':
    main()
