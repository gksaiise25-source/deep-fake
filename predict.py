"""
Prediction Module for Deepfake Detection

This module handles:
- Making predictions on single images
- Making predictions on video frames
- Visualization of results
"""

import cv2
import numpy as np
import os
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from data_preprocessing import FaceDetector, ImagePreprocessor, VideoProcessor
from model import DeepfakeDetectionModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepfakePredictor:
    """
    Predictor class for deepfake detection
    """
    
    def __init__(self, model_path: str, model_type: str = 'efficientnet',
                 input_shape: Tuple[int, int] = (256, 256)):
        """
        Initialize predictor with a trained model
        
        Args:
            model_path (str): Path to saved model (.h5 file)
            model_type (str): Type of model
            input_shape (tuple): Input image shape
        """
        self.model_path = model_path
        self.model_type = model_type
        self.input_shape = input_shape
        
        # Load model
        self.model_obj = DeepfakeDetectionModel(model_type=model_type, input_shape=input_shape)
        try:
            self.model_obj.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Initialize preprocessing tools
        self.face_detector = FaceDetector(method='opencv')
        self.image_processor = ImagePreprocessor(target_size=input_shape)
        self.video_processor = VideoProcessor(target_size=input_shape)
    
    def predict_image(self, image_path: str, detect_face: bool = True) -> Dict:
        """
        Predict on a single image
        
        Args:
            image_path (str): Path to image file
            detect_face (bool): Whether to detect and extract face
            
        Returns:
            Dictionary with prediction results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image_rgb.copy()
        
        # Detect face if requested
        face_detected = False
        if detect_face:
            faces = self.face_detector.detect_faces(image)
            if faces:
                face = self.face_detector.extract_face(image, faces[0])
                if face is not None:
                    image_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_detected = True
            else:
                logger.warning("No face detected in image, using full image")
        
        # Preprocess
        preprocessed = self.image_processor.preprocess(image_rgb)
        
        # Add batch dimension
        input_data = np.expand_dims(preprocessed, axis=0)
        
        # Predict
        prediction = self.model_obj.get_model().predict(input_data, verbose=0)[0][0]
        
        # Prepare results
        is_fake = prediction > 0.5
        confidence = max(prediction, 1 - prediction) * 100
        
        result = {
            'image_path': image_path,
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'raw_prediction': float(prediction),
            'label': 'FAKE' if is_fake else 'REAL',
            'face_detected': face_detected,
            'original_image': original_image,
            'processed_image': preprocessed
        }
        
        logger.info(f"Prediction for {image_path}: {result['label']} ({confidence:.2f}%)")
        
        return result
    
    def predict_video(self, video_path: str, max_frames: int = 30,
                     detect_face: bool = True) -> Dict:
        """
        Predict on a video
        
        Args:
            video_path (str): Path to video file
            max_frames (int): Maximum frames to process
            detect_face (bool): Whether to detect and extract faces
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Processing video: {video_path}")
        
        # Extract frames
        frames = self.video_processor.extract_frames(video_path, max_frames=max_frames)
        
        if not frames:
            logger.error("No frames extracted from video")
            return None
        
        predictions = []
        
        for frame_idx, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face if requested
            if detect_face:
                faces = self.face_detector.detect_faces(frame)
                if faces:
                    face = self.face_detector.extract_face(frame, faces[0])
                    if face is not None:
                        frame_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            preprocessed = self.image_processor.preprocess(frame_rgb)
            
            # Predict
            input_data = np.expand_dims(preprocessed, axis=0)
            prediction = self.model_obj.get_model().predict(input_data, verbose=0)[0][0]
            
            predictions.append({
                'frame': frame_idx,
                'raw_prediction': float(prediction),
                'is_fake': prediction > 0.5,
                'confidence': float(max(prediction, 1 - prediction) * 100)
            })
        
        # Aggregate results
        is_fake_list = [p['is_fake'] for p in predictions]
        confidence_list = [p['confidence'] for p in predictions]
        
        overall_is_fake = sum(is_fake_list) > len(is_fake_list) / 2
        overall_confidence = np.mean(confidence_list)
        
        result = {
            'video_path': video_path,
            'total_frames_processed': len(frames),
            'is_fake': overall_is_fake,
            'confidence': float(overall_confidence),
            'label': 'FAKE' if overall_is_fake else 'REAL',
            'frame_predictions': predictions,
            'fake_frame_percentage': float(sum(is_fake_list) / len(is_fake_list) * 100)
        }
        
        logger.info(f"Video prediction: {result['label']} ({overall_confidence:.2f}%)")
        logger.info(f"Fake frames: {sum(is_fake_list)}/{len(frames)} ({result['fake_frame_percentage']:.1f}%)")
        
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict on a batch of images
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            if result:
                results.append(result)
        
        return results
    
    def visualize_prediction(self, result: Dict, save_path: Optional[str] = None):
        """
        Visualize prediction result
        
        Args:
            result (Dict): Prediction result from predict_image
            save_path (str): Path to save visualization (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Processed image
        axes[1].imshow(result['processed_image'])
        title = f"{result['label']}\nConfidence: {result['confidence']:.2f}%"
        color = 'red' if result['is_fake'] else 'green'
        axes[1].set_title(title, color=color, fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_video_results(self, result: Dict, save_path: Optional[str] = None):
        """
        Visualize video prediction results
        
        Args:
            result (Dict): Prediction result from predict_video
            save_path (str): Path to save visualization (optional)
        """
        predictions = result['frame_predictions']
        frames = [p['frame'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        is_fakes = [p['is_fake'] for p in predictions]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Confidence plot
        colors = ['red' if f else 'green' for f in is_fakes]
        axes[0].bar(frames, confidences, color=colors, alpha=0.7)
        axes[0].axhline(y=50, color='black', linestyle='--', label='Decision Threshold')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Confidence (%)')
        axes[0].set_title('Per-Frame Prediction Confidence')
        axes[0].legend()
        axes[0].set_ylim([0, 100])
        
        # Summary
        axes[1].axis('off')
        summary_text = f"""
        Video: {result['video_path']}
        Overall Prediction: {result['label']}
        Overall Confidence: {result['confidence']:.2f}%
        Total Frames Processed: {result['total_frames_processed']}
        Fake Frames: {result['fake_frame_percentage']:.1f}%
        """
        axes[1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict using Deepfake Detection Model')
    parser.add_argument('--model', type=str, default='models/deepfake_efficientnet_final.h5',
                       help='Path to model file')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--video', type=str, help='Path to video for prediction')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    parser.add_argument('--no-face-detection', action='store_true',
                       help='Disable face detection')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.info("Please train a model first using train.py")
        return
    
    # Initialize predictor
    predictor = DeepfakePredictor(args.model)
    
    # Make predictions
    if args.image:
        if not os.path.exists(args.image):
            logger.error(f"Image file not found: {args.image}")
            return
        
        result = predictor.predict_image(args.image, detect_face=not args.no_face_detection)
        if result:
            print(f"\nPrediction Result:")
            print(f"Label: {result['label']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Face Detected: {result['face_detected']}")
            
            if args.output:
                predictor.visualize_prediction(result, save_path=args.output)
            else:
                predictor.visualize_prediction(result)
    
    elif args.video:
        if not os.path.exists(args.video):
            logger.error(f"Video file not found: {args.video}")
            return
        
        result = predictor.predict_video(args.video, detect_face=not args.no_face_detection)
        if result:
            print(f"\nVideo Prediction Result:")
            print(f"Label: {result['label']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Fake Frames: {result['fake_frame_percentage']:.1f}%")
            
            if args.output:
                predictor.visualize_video_results(result, save_path=args.output)
            else:
                predictor.visualize_video_results(result)
    
    else:
        print("Please provide --image or --video argument")
        parser.print_help()


if __name__ == '__main__':
    main()
