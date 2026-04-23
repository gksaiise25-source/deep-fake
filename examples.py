"""
Advanced Examples for Deepfake Detection System

This file contains advanced usage examples
"""

import numpy as np
import matplotlib.pyplot as plt
from predict import DeepfakePredictor
from train import ModelTrainer
from data_preprocessing import DatasetLoader, ImagePreprocessor
import glob


# Example 1: Custom Prediction Pipeline
def example_custom_prediction():
    """Custom prediction with preprocessing and visualization"""
    
    print("\n" + "="*60)
    print("Example 1: Custom Prediction Pipeline")
    print("="*60)
    
    predictor = DeepfakePredictor('models/deepfake_efficientnet_final.h5')
    
    # Predict on multiple images
    image_paths = glob.glob('data/test/*.jpg')[:5]  # First 5 images
    
    for image_path in image_paths:
        result = predictor.predict_image(image_path)
        
        if result:
            print(f"\n{image_path}:")
            print(f"  Prediction: {result['label']}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print(f"  Raw Score: {result['raw_prediction']:.4f}")


# Example 2: Batch Processing with Summary
def example_batch_processing():
    """Process multiple images and generate summary report"""
    
    print("\n" + "="*60)
    print("Example 2: Batch Processing with Summary")
    print("="*60)
    
    predictor = DeepfakePredictor('models/deepfake_efficientnet_final.h5')
    
    real_images = glob.glob('data/train_real/*.jpg')[:10]
    fake_images = glob.glob('data/train_fake/*.jpg')[:10]
    
    results = predictor.predict_batch(real_images + fake_images)
    
    # Generate statistics
    total = len(results)
    correct = sum(1 for r in results if r['is_fake'] == r['image_path'] in fake_images)
    
    print(f"\nBatch Statistics:")
    print(f"Total images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    
    # Group by confidence
    high_conf = sum(1 for r in results if r['confidence'] > 90)
    medium_conf = sum(1 for r in results if 70 <= r['confidence'] <= 90)
    low_conf = sum(1 for r in results if r['confidence'] < 70)
    
    print(f"\nConfidence Distribution:")
    print(f"High (>90%): {high_conf}")
    print(f"Medium (70-90%): {medium_conf}")
    print(f"Low (<70%): {low_conf}")


# Example 3: Video Analysis Deep Dive
def example_video_analysis():
    """Detailed video analysis with frame-by-frame breakdown"""
    
    print("\n" + "="*60)
    print("Example 3: Detailed Video Analysis")
    print("="*60)
    
    predictor = DeepfakePredictor('models/deepfake_efficientnet_final.h5')
    
    video_path = 'data/test_video.mp4'
    result = predictor.predict_video(video_path, max_frames=30)
    
    if result:
        print(f"\nVideo: {video_path}")
        print(f"Overall Prediction: {result['label']}")
        print(f"Overall Confidence: {result['confidence']:.2f}%")
        print(f"Total Frames: {result['total_frames_processed']}")
        print(f"Fake Frames: {result['fake_frame_percentage']:.1f}%")
        
        # Find frames with highest fake confidence
        predictions = sorted(result['frame_predictions'], 
                           key=lambda x: x['confidence'], 
                           reverse=True)
        
        print(f"\nTop 5 most suspicious frames:")
        for i, pred in enumerate(predictions[:5], 1):
            print(f"{i}. Frame {pred['frame']}: {pred['confidence']:.2f}% fake")


# Example 4: Model Comparison
def example_model_comparison():
    """Compare predictions from different models"""
    
    print("\n" + "="*60)
    print("Example 4: Model Comparison")
    print("="*60)
    
    models = [
        ('models/deepfake_efficientnet_final.h5', 'EfficientNet'),
        ('models/deepfake_resnet_final.h5', 'ResNet50'),
        ('models/deepfake_xception_final.h5', 'Xception'),
    ]
    
    test_image = 'data/test_real/sample.jpg'
    
    print(f"\nTesting image: {test_image}\n")
    print(f"{'Model':<15} {'Prediction':<10} {'Confidence':<12} {'Raw Score':<10}")
    print("-" * 50)
    
    predictions = {}
    
    for model_path, model_name in models:
        try:
            predictor = DeepfakePredictor(model_path)
            result = predictor.predict_image(test_image)
            
            if result:
                predictions[model_name] = result
                print(f"{model_name:<15} {result['label']:<10} {result['confidence']:>6.2f}% {result['raw_prediction']:>8.4f}")
        except Exception as e:
            print(f"{model_name:<15} Error: {str(e)[:25]}")
    
    # Consensus prediction
    if len(predictions) > 0:
        votes = sum(1 for r in predictions.values() if r['is_fake'])
        consensus = 'FAKE' if votes > len(predictions)/2 else 'REAL'
        print(f"\nConsensus: {consensus} ({votes}/{len(predictions)} models agree)")


# Example 5: Training with Custom Configuration
def example_custom_training():
    """Train with custom configuration"""
    
    print("\n" + "="*60)
    print("Example 5: Custom Training Configuration")
    print("="*60)
    
    # Load dataset
    loader = DatasetLoader(target_size=(256, 256))
    X, y = loader.load_dataset('data/train_real', 'data/train_fake', max_images=100)
    
    if len(X) == 0:
        print("No data found. Create sample data first: python train.py --create-sample")
        return
    
    # Create custom trainer
    trainer = ModelTrainer(model_type='efficientnet')
    
    # Prepare data with custom split
    data = trainer.prepare_data(X, y, test_size=0.15, val_size=0.15)
    
    print(f"\nDataset Summary:")
    print(f"Training: {len(data['X_train'])} images")
    print(f"Validation: {len(data['X_val'])} images")
    print(f"Testing: {len(data['X_test'])} images")
    
    # Optional: Train (comment out if not needed)
    # trainer.train(data, epochs=10)
    # results = trainer.evaluate(data['X_test'], data['y_test'])


# Example 6: Visualization Examples
def example_visualization():
    """Create custom visualizations"""
    
    print("\n" + "="*60)
    print("Example 6: Visualization Examples")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    epochs = np.arange(1, 51)
    train_loss = np.exp(-epochs/10) + np.random.normal(0, 0.01, 50)
    val_loss = np.exp(-epochs/10) + np.random.normal(0, 0.02, 50)
    train_acc = 1 - np.exp(-epochs/20) + np.random.normal(0, 0.01, 50)
    val_acc = 1 - np.exp(-epochs/25) + np.random.normal(0, 0.015, 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(epochs, train_loss, label='Training Loss', alpha=0.8)
    axes[0, 0].plot(epochs, val_loss, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, train_acc, label='Training Accuracy', alpha=0.8)
    axes[0, 1].plot(epochs, val_acc, label='Validation Accuracy', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = np.array([[450, 50], [30, 470]])
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
        disp.plot(ax=axes[1, 0], cmap='Blues')
    except:
        axes[1, 0].imshow(cm, cmap='Blues')
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_xticklabels(['Real', 'Fake'])
        axes[1, 0].set_yticklabels(['Real', 'Fake'])
        axes[1, 0].set_title('Confusion Matrix')
    
    # Metrics bar chart
    metrics = np.array([0.92, 0.91, 0.93, 0.92])
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    axes[1, 1].bar(labels, metrics, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (label, value) in enumerate(zip(labels, metrics)):
        axes[1, 1].text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('outputs/training_analysis.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to outputs/training_analysis.png")
    plt.show()


# Example 7: Real-time Monitoring
def example_monitoring():
    """Monitor model predictions in real-time"""
    
    print("\n" + "="*60)
    print("Example 7: Real-time Monitoring")
    print("="*60)
    
    import cv2
    import time
    
    print("\nNote: This example requires a webcam")
    print("Uncomment the code below to use it")
    print("""
    # Requires: pip install opencv-python
    # May require: pip install opencv-contrib-python
    
    predictor = DeepfakePredictor('models/deepfake_efficientnet_final.h5')
    cap = cv2.VideoCapture(0)  # Webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        processor = ImagePreprocessor((256, 256))
        processed = processor.preprocess(frame)
        
        # Predict
        input_data = np.expand_dims(processed, axis=0)
        prediction = predictor.model_obj.get_model().predict(input_data, verbose=0)[0][0]
        
        # Display result
        label = 'FAKE' if prediction > 0.5 else 'REAL'
        confidence = max(prediction, 1-prediction) * 100
        color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
        
        cv2.putText(frame, f'{label} ({confidence:.1f}%)', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Deepfake Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    """)


# Example 8: Export Predictions to CSV
def example_export_predictions():
    """Export predictions to CSV file"""
    
    print("\n" + "="*60)
    print("Example 8: Export Predictions to CSV")
    print("="*60)
    
    import csv
    import os
    from datetime import datetime
    
    predictor = DeepfakePredictor('models/deepfake_efficientnet_final.h5')
    
    image_paths = glob.glob('data/test/*.jpg')[:10]
    results = predictor.predict_batch(image_paths)
    
    # Export to CSV
    csv_file = f'outputs/predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    os.makedirs('outputs', exist_ok=True)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'prediction', 'confidence', 'raw_score'])
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'image_path': result['image_path'],
                'prediction': result['label'],
                'confidence': f"{result['confidence']:.2f}",
                'raw_score': f"{result['raw_prediction']:.4f}"
            })
    
    print(f"Predictions exported to {csv_file}")


def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION SYSTEM - ADVANCED EXAMPLES")
    print("="*70)
    
    examples = [
        ("Custom Prediction Pipeline", example_custom_prediction),
        ("Batch Processing", example_batch_processing),
        ("Video Analysis", example_video_analysis),
        ("Model Comparison", example_model_comparison),
        ("Custom Training", example_custom_training),
        ("Visualization", example_visualization),
        ("Real-time Monitoring", example_monitoring),
        ("Export Predictions", example_export_predictions),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nTo run an example, uncomment it in main() and run:")
    print("python examples.py")
    
    # Uncomment examples to run:
    # example_custom_prediction()
    # example_batch_processing()
    # example_video_analysis()
    # example_model_comparison()
    # example_custom_training()
    # example_visualization()
    # example_monitoring()
    # example_export_predictions()


if __name__ == '__main__':
    main()
