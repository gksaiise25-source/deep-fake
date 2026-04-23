"""
Quick Prediction Script - Easy to use!

Simply run:
    python quick_predict.py --image "path/to/your/image.jpg"
"""

import argparse
import numpy as np
import cv2
from model_sklearn import DeepfakeDetectionModel
from data_preprocessing import ImagePreprocessor
import os

def main():
    parser = argparse.ArgumentParser(description='Quick Deepfake Detection Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔍 Deepfake Detection Prediction")
    print(f"{'='*60}\n")
    
    try:
        # Load the trained model
        print("📦 Loading trained model...")
        model = DeepfakeDetectionModel()
        model.load('models/deepfake_random_forest_sklearn.pkl')
        print("✅ Model loaded successfully!\n")
        
        # Load and preprocess image
        print(f"📷 Loading image: {args.image}")
        img = cv2.imread(args.image)
        
        if img is None:
            print(f"❌ Error: Could not read image file")
            return
        
        # Preprocess
        print("🔄 Preprocessing image...")
        processor = ImagePreprocessor(target_size=(256, 256))
        processed = processor.preprocess(img)
        
        # Make prediction
        print("🤖 Making prediction...\n")
        input_data = np.expand_dims(processed, axis=0)
        
        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Extract confidence
        is_fake = prediction == 1
        confidence = max(probabilities) * 100
        
        # Display results
        print(f"{'='*60}")
        if is_fake:
            print(f"⚠️  RESULT: FAKE / DEEPFAKE DETECTED")
            print(f"{'='*60}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"(This image appears to be AI-generated or manipulated)")
        else:
            print(f"✅ RESULT: REAL / AUTHENTIC")
            print(f"{'='*60}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"(This image appears to be genuine)")
        
        print(f"\nRaw Probabilities:")
        print(f"  Real:      {probabilities[0]*100:.2f}%")
        print(f"  Fake:      {probabilities[1]*100:.2f}%")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
