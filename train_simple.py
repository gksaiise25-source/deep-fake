"""
Simplified Training Script for Deepfake Detection

Works with scikit-learn and Python 3.14+
"""

import numpy as np
import os
import argparse
import logging
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from data_preprocessing import DatasetLoader
from model_sklearn import DeepfakeDetectionModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTrainer:
    """
    Simplified trainer for sklearn models
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize trainer
        
        Args:
            model_type (str): Type of model to train
        """
        self.model_type = model_type
        self.model = DeepfakeDetectionModel(model_type=model_type)
        self.test_results = None
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, val_size: float = 0.1,
                    shuffle: bool = True, seed: int = 42):
        """
        Prepare train, validation, and test datasets
        
        Args:
            X (np.ndarray): Images
            y (np.ndarray): Labels
            test_size (float): Fraction of data for testing
            val_size (float): Fraction of training data for validation
            shuffle (bool): Whether to shuffle data
            seed (int): Random seed
            
        Returns:
            Dictionary with train, val, test splits
        """
        logger.info(f"Preparing data with test_size={test_size}, val_size={val_size}")
        
        # For small datasets, use minimum test/val sizes
        min_test = max(2, int(len(y) * 0.2))
        if len(y) < 10:
            test_size = min_test / len(y)
            val_size = 0.0  # No validation for very small datasets
        
        # Split into train+val and test
        stratify_arg = y if len(np.unique(y)) == 2 and len(y) > 4 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            shuffle=shuffle,
            stratify=stratify_arg
        )
        
        # Split train into train and val only if we have enough data
        if val_size > 0 and len(X_train) > 4:
            stratify_arg_val = y_train if len(np.unique(y_train)) == 2 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_size,
                random_state=seed,
                shuffle=shuffle,
                stratify=stratify_arg_val
            )
        else:
            X_val, y_val = X_train[:0], y_train[:0]  # Empty validation set
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Train distribution: {np.sum(y_train==0)} real, {np.sum(y_train==1)} fake")
        if len(y_val) > 0:
            logger.info(f"Val distribution: {np.sum(y_val==0)} real, {np.sum(y_val==1)} fake")
        logger.info(f"Test distribution: {np.sum(y_test==0)} real, {np.sum(y_test==1)} fake")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def train(self, data_dict: dict, model_name: str = 'deepfake_detector'):
        """
        Train the model
        
        Args:
            data_dict (dict): Dictionary with X_train, y_train, X_val, y_val
            model_name (str): Name for saved model
        """
        logger.info(f"Starting training with {self.model_type}")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        logger.info("Training completed")
        
        # Save model
        model_path = f'models/{model_name}_sklearn.pkl'
        self.model.save(model_path)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test set
        
        Args:
            X_test (np.ndarray): Test images
            y_test (np.ndarray): Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, 
                                            target_names=['Real', 'Fake'],
                                            output_dict=True)
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"AUC: {auc:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Real', 'Fake'])}")
        
        self.test_results = results
        return results
    
    def save_results(self, results: dict, filepath: str = 'outputs/results.json'):
        """
        Save evaluation results to JSON
        
        Args:
            results (dict): Results dictionary
            filepath (str): Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {filepath}")


def create_sample_dataset():
    """
    Create a small sample dataset for testing
    """
    logger.info("Creating sample dataset...")
    
    # Create sample directories
    os.makedirs('data/train_real', exist_ok=True)
    os.makedirs('data/train_fake', exist_ok=True)
    
    # Generate random images as samples (256x256x3)
    np.random.seed(42)
    
    # Create 20 sample real images
    for i in range(20):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(f'data/train_real/sample_real_{i:03d}.jpg', img)
    
    # Create 20 sample fake images (with some variation)
    for i in range(20):
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(f'data/train_fake/sample_fake_{i:03d}.jpg', img)
    
    logger.info("Sample dataset created in data/train_real and data/train_fake")


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boost'],
                       help='Model type to train')
    parser.add_argument('--real-dir', type=str, default='data/train_real',
                       help='Directory with real images')
    parser.add_argument('--fake-dir', type=str, default='data/train_fake',
                       help='Directory with fake images')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images per class to load')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample dataset for testing')
    
    args = parser.parse_args()
    
    # Create sample dataset if requested
    if args.create_sample:
        create_sample_dataset()
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DatasetLoader(target_size=(256, 256))
    X, y = dataset_loader.load_dataset(args.real_dir, args.fake_dir, 
                                       max_images=args.max_images)
    
    if len(X) == 0:
        logger.error("No data loaded. Please provide valid dataset directories.")
        logger.info("Try running: python train_simple.py --create-sample")
        return
    
    # Initialize trainer
    trainer = SimpleTrainer(model_type=args.model)
    
    # Prepare data
    data_dict = trainer.prepare_data(X, y)
    
    # Train model
    trainer.train(data_dict, model_name=f'deepfake_{args.model}')
    
    # Evaluate
    results = trainer.evaluate(data_dict['X_test'], data_dict['y_test'])
    
    # Save results
    trainer.save_results(results)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
