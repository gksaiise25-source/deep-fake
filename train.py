"""
Training Script for Deepfake Detection Model

This script handles:
- Dataset loading and preparation
- Model training with various configurations
- Validation and evaluation
- Model saving
"""

import numpy as np
import os
import argparse
import logging
from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from data_preprocessing import DatasetLoader
from model import DeepfakeDetectionModel, get_callbacks

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer class for deepfake detection model
    """
    
    def __init__(self, model_type: str = 'efficientnet', input_shape: tuple = (256, 256, 3)):
        """
        Initialize trainer
        
        Args:
            model_type (str): Type of model to train
            input_shape (tuple): Input image shape
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = DeepfakeDetectionModel(model_type=model_type, input_shape=input_shape)
        self.model.compile()
        self.history = None
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
        
        # Split into train+val and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            shuffle=shuffle,
            stratify=y
        )
        
        # Split train into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=seed,
            shuffle=shuffle,
            stratify=y_train
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Train distribution: {np.sum(y_train==0)} real, {np.sum(y_train==1)} fake")
        logger.info(f"Val distribution: {np.sum(y_val==0)} real, {np.sum(y_val==1)} fake")
        logger.info(f"Test distribution: {np.sum(y_test==0)} real, {np.sum(y_test==1)} fake")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def get_data_augmentation(self) -> ImageDataGenerator:
        """
        Get data augmentation generator
        
        Returns:
            ImageDataGenerator for training data
        """
        augmentation = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.15,
            fill_mode='nearest'
        )
        
        logger.info("Data augmentation configured")
        return augmentation
    
    def train(self, data_dict: dict, epochs: int = 50, batch_size: int = 32,
             use_augmentation: bool = True, model_name: str = 'deepfake_detector'):
        """
        Train the model
        
        Args:
            data_dict (dict): Dictionary with X_train, y_train, X_val, y_val
            epochs (int): Number of epochs
            batch_size (int): Batch size
            use_augmentation (bool): Whether to use data augmentation
            model_name (str): Name for saved model
        """
        logger.info(f"Starting training for {epochs} epochs with batch_size={batch_size}")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        callbacks = get_callbacks(model_name=model_name)
        
        if use_augmentation:
            augmentation = self.get_data_augmentation()
            
            self.history = self.model.get_model().fit(
                augmentation.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.get_model().fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        logger.info("Training completed")
        
        # Save final model
        final_model_path = f'models/{model_name}_final.h5'
        self.model.save(final_model_path)
    
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
        y_pred_prob = self.model.get_model().predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        
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
    
    def save_training_history(self, filepath: str = 'outputs/training_history.json'):
        """
        Save training history to JSON
        
        Args:
            filepath (str): Path to save history
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        history_dict = {
            'accuracy': [float(x) for x in self.history.history.get('accuracy', [])],
            'loss': [float(x) for x in self.history.history.get('loss', [])],
            'val_accuracy': [float(x) for x in self.history.history.get('val_accuracy', [])],
            'val_loss': [float(x) for x in self.history.history.get('val_loss', [])],
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        logger.info(f"Training history saved to {filepath}")


def create_sample_dataset():
    """
    Create a small sample dataset for testing
    Use this when you don't have the actual dataset
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
        cv2_available = False
        try:
            import cv2
            cv2.imwrite(f'data/train_real/sample_real_{i:03d}.jpg', img)
            cv2_available = True
        except:
            # If cv2 not available, use PIL
            from PIL import Image
            Image.fromarray(img).save(f'data/train_real/sample_real_{i:03d}.jpg')
    
    # Create 20 sample fake images
    for i in range(20):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        try:
            import cv2
            cv2.imwrite(f'data/train_fake/sample_fake_{i:03d}.jpg', img)
        except:
            from PIL import Image
            Image.fromarray(img).save(f'data/train_fake/sample_fake_{i:03d}.jpg')
    
    logger.info("Sample dataset created in data/train_real and data/train_fake")


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--model', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'xception', 'custom'],
                       help='Model type to train')
    parser.add_argument('--real-dir', type=str, default='data/train_real',
                       help='Directory with real images')
    parser.add_argument('--fake-dir', type=str, default='data/train_fake',
                       help='Directory with fake images')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images per class to load')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
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
        logger.info("Try running: python train.py --create-sample")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(model_type=args.model)
    
    # Prepare data
    data_dict = trainer.prepare_data(X, y)
    
    # Train model
    trainer.train(data_dict, epochs=args.epochs, batch_size=args.batch_size,
                 use_augmentation=not args.no_augmentation,
                 model_name=f'deepfake_{args.model}')
    
    # Evaluate
    results = trainer.evaluate(data_dict['X_test'], data_dict['y_test'])
    
    # Save results
    trainer.save_results(results)
    trainer.save_training_history()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
