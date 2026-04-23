"""
Model Validation and Metrics Script

Evaluates model performance with:
- Confusion matrix
- ROC curves
- Precision, Recall, F1-score
- Detailed classification reports

Run with:
    python validate_model.py
"""

import os
import sys
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    auc, roc_auc_score, precision_recall_curve, f1_score,
    accuracy_score, precision_score, recall_score
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelValidator:
    """Validate and analyze model performance"""
    
    def __init__(self, model, preprocessor, feature_extractor):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
    
    def predict_directory(self, image_dir, label):
        """Get predictions for all images in directory"""
        predictions = []
        probabilities = []
        true_labels = []
        
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for filename in image_files:
            filepath = os.path.join(image_dir, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                continue
            
            # Preprocess
            if image.max() > 1:
                image = image / 255.0
            
            processed = self.preprocessor.preprocess(image)
            features = self.feature_extractor.extract_features(processed)
            features = features.reshape(1, -1)
            
            # Predict
            pred = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            
            predictions.append(pred)
            probabilities.append(proba[1])  # Confidence for fake
            true_labels.append(label)
        
        return np.array(predictions), np.array(probabilities), np.array(true_labels)
    
    def evaluate(self, real_dir, fake_dir, output_dir='outputs'):
        """Evaluate model on test data"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60 + "\n")
        
        print("Evaluating REAL images...")
        real_preds, real_probs, real_labels = self.predict_directory(real_dir, 0)
        
        print("Evaluating FAKE images...")
        fake_preds, fake_probs, fake_labels = self.predict_directory(fake_dir, 1)
        
        # Combine results
        y_true = np.concatenate([real_labels, fake_labels])
        y_pred = np.concatenate([real_preds, fake_preds])
        y_proba = np.concatenate([real_probs, fake_probs])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except:
            roc_auc = 0.0
        
        # Print metrics
        print("\n" + "="*60)
        print("📈 PERFORMANCE METRICS")
        print("="*60)
        print(f"✅ Accuracy:  {accuracy:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall:    {recall:.4f}")
        print(f"✅ F1-Score:  {f1:.4f}")
        print(f"✅ ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0, 0]}")
        print(f"   False Positives: {cm[0, 1]}")
        print(f"   False Negatives: {cm[1, 0]}")
        print(f"   True Positives:  {cm[1, 1]}")
        
        # Classification Report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, output_dir)
        
        # Plot ROC curve
        if len(np.unique(y_true)) > 1:
            self._plot_roc_curve(y_true, y_proba, output_dir)
        
        # Plot precision-recall curve
        if len(np.unique(y_true)) > 1:
            self._plot_precision_recall(y_true, y_proba, output_dir)
        
        # Save results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        joblib.dump(results, os.path.join(output_dir, 'validation_results.pkl'))
        
        print(f"\n✅ Results saved to {output_dir}")
        
        return results
    
    def _plot_confusion_matrix(self, cm, output_dir):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        logger.info(f"Saved confusion matrix plot")
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_proba, output_dir):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
        logger.info(f"Saved ROC curve plot")
        plt.close()
    
    def _plot_precision_recall(self, y_true, y_proba, output_dir):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall.png'), dpi=150)
        logger.info(f"Saved precision-recall curve plot")
        plt.close()


def main():
    """Main validation pipeline"""
    from model_advanced import AdvancedDeepfakeDetectionModel
    from data_preprocessing import ImagePreprocessor
    from advanced_features import AdvancedFeatureExtractor
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load model
    model = AdvancedDeepfakeDetectionModel()
    model_path = os.path.join(base_dir, 'models', 'deepfake_advanced_ra.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Run: python train_advanced.py")
        sys.exit(1)
    
    model.load(model_path)
    
    # Validator
    validator = ModelValidator(model.model, model.preprocessor, model.feature_extractor)
    
    # Test data (use training data for now)
    real_dir = os.path.join(base_dir, 'data', 'train_real')
    fake_dir = os.path.join(base_dir, 'data', 'train_fake')
    output_dir = os.path.join(base_dir, 'outputs')
    
    # Evaluate
    results = validator.evaluate(real_dir, fake_dir, output_dir)


if __name__ == "__main__":
    main()
