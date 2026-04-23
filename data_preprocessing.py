"""
Data Preprocessing Module for Deepfake Detection

This module handles:
- Face detection from images and videos
- Frame extraction from videos
- Image resizing and normalization
- Dataset preparation
"""

import cv2
import numpy as np
import os
from pathlib import Path
import logging
from typing import Tuple, List, Optional

# Optional import - dlib is not required
try:
    import dlib
except ImportError:
    dlib = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detection class using OpenCV and dlib for accurate face detection
    """
    
    def __init__(self, method='opencv'):
        """
        Initialize face detector
        
        Args:
            method (str): 'opencv' or 'dlib' for face detection
        """
        self.method = method
        
        if method == 'dlib' and dlib is None:
            logger.warning("dlib not available. Falling back to OpenCV")
            method = 'opencv'
            self.method = 'opencv'
        
        if method == 'opencv':
            # Load pre-trained Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        elif method == 'dlib':
            # Load dlib face detector
            try:
                self.detector = dlib.get_frontal_face_detector()
            except Exception as e:
                logger.warning(f"dlib not available: {e}. Falling back to OpenCV")
                self.method = 'opencv'
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List of face bounding boxes (x, y, width, height)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.method == 'opencv':
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        
        elif self.method == 'dlib':
            dets = self.detector(gray, 1)
            faces = []
            for det in dets:
                x, y, w, h = det.left(), det.top(), det.width(), det.height()
                faces.append((x, y, w, h))
            return faces
    
    def extract_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int], 
                    padding: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract face region from image with padding
        
        Args:
            image (np.ndarray): Input image
            face_box (tuple): Face bounding box (x, y, w, h)
            padding (float): Padding around face (0.1 = 10%)
            
        Returns:
            Extracted face image or None
        """
        x, y, w, h = face_box
        
        # Add padding
        pad_x = int(w * padding / 2)
        pad_y = int(h * padding / 2)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)
        
        if x2 - x1 > 0 and y2 - y1 > 0:
            return image[y1:y2, x1:x2]
        return None


class ImagePreprocessor:
    """
    Image preprocessing and normalization
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize preprocessor
        
        Args:
            target_size (tuple): Target image size (height, width)
        """
        self.target_size = target_size
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: resize and normalize
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Preprocessed image (normalized to [0, 1])
        """
        # Resize image
        resized = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # Convert to float and normalize
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def batch_preprocess(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess batch of images
        
        Args:
            images (List[np.ndarray]): List of input images
            
        Returns:
            Batch of preprocessed images
        """
        batch = np.array([self.preprocess(img) for img in images])
        return batch


class VideoProcessor:
    """
    Video processing and frame extraction
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256), sample_rate: int = 5):
        """
        Initialize video processor
        
        Args:
            target_size (tuple): Target frame size
            sample_rate (int): Sample every nth frame (e.g., 5 = every 5th frame)
        """
        self.target_size = target_size
        self.sample_rate = sample_rate
        self.face_detector = FaceDetector(method='opencv')
        self.image_processor = ImagePreprocessor(target_size)
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            video_path (str): Path to video file
            max_frames (int): Maximum number of frames to extract (None = all)
            
        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return frames
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_count % self.sample_rate == 0:
                frames.append(frame)
            
            frame_count += 1
            
            if max_frames and len(frames) >= max_frames:
                break
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def extract_faces_from_video(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract face regions from video
        
        Args:
            video_path (str): Path to video file
            max_frames (int): Maximum number of frames to process
            
        Returns:
            List of face images
        """
        frames = self.extract_frames(video_path, max_frames)
        faces = []
        
        for frame in frames:
            face_boxes = self.face_detector.detect_faces(frame)
            
            for face_box in face_boxes:
                face = self.face_detector.extract_face(frame, face_box)
                if face is not None:
                    preprocessed_face = self.image_processor.preprocess(face)
                    faces.append(preprocessed_face)
        
        return faces


class DatasetLoader:
    """
    Load and prepare dataset from directories
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize dataset loader
        
        Args:
            target_size (tuple): Target image size
        """
        self.target_size = target_size
        self.face_detector = FaceDetector(method='opencv')
        self.image_processor = ImagePreprocessor(target_size)
        self.video_processor = VideoProcessor(target_size)
    
    def load_images_from_directory(self, directory: str, label: int, max_images: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load images from directory
        
        Args:
            directory (str): Path to directory with images
            label (int): Label for images (0=real, 1=fake)
            max_images (int): Maximum number of images to load
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return images, labels
        
        image_files = [f for f in os.listdir(directory) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        if max_images:
            image_files = image_files[:max_images]
        
        for filename in image_files:
            filepath = os.path.join(directory, filename)
            try:
                image = cv2.imread(filepath)
                if image is not None:
                    # Detect and extract face
                    faces = self.face_detector.detect_faces(image)
                    
                    if faces:
                        face = self.face_detector.extract_face(image, faces[0])
                        if face is not None:
                            preprocessed = self.image_processor.preprocess(face)
                            images.append(preprocessed)
                            labels.append(label)
                    else:
                        # If no face detected, use full image
                        preprocessed = self.image_processor.preprocess(image)
                        images.append(preprocessed)
                        labels.append(label)
            except Exception as e:
                logger.warning(f"Error loading image {filename}: {e}")
        
        logger.info(f"Loaded {len(images)} images from {directory}")
        return images, labels
    
    def load_videos_from_directory(self, directory: str, label: int, 
                                  max_videos: Optional[int] = None,
                                  frames_per_video: int = 10) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load faces from videos in directory
        
        Args:
            directory (str): Path to directory with videos
            label (int): Label for videos (0=real, 1=fake)
            max_videos (int): Maximum number of videos to process
            frames_per_video (int): Frames to extract per video
            
        Returns:
            Tuple of (faces, labels)
        """
        faces = []
        labels = []
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return faces, labels
        
        video_files = [f for f in os.listdir(directory) 
                      if os.path.splitext(f)[1].lower() in video_extensions]
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        for filename in video_files:
            filepath = os.path.join(directory, filename)
            try:
                video_faces = self.video_processor.extract_faces_from_video(
                    filepath, 
                    max_frames=frames_per_video
                )
                faces.extend(video_faces)
                labels.extend([label] * len(video_faces))
            except Exception as e:
                logger.warning(f"Error processing video {filename}: {e}")
        
        logger.info(f"Loaded {len(faces)} faces from {len(video_files)} videos")
        return faces, labels
    
    def load_dataset(self, real_dir: str, fake_dir: str, 
                    max_images: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load complete dataset from real and fake directories
        
        Args:
            real_dir (str): Directory with real images
            fake_dir (str): Directory with fake images
            max_images (int): Max images per class
            
        Returns:
            Tuple of (X, y) - images and labels
        """
        logger.info("Loading dataset...")
        
        real_images, real_labels = self.load_images_from_directory(real_dir, label=0, max_images=max_images)
        fake_images, fake_labels = self.load_images_from_directory(fake_dir, label=1, max_images=max_images)
        
        X = np.array(real_images + fake_images)
        y = np.array(real_labels + fake_labels)
        
        logger.info(f"Dataset prepared: {len(X)} images, Real: {sum(y==0)}, Fake: {sum(y==1)}")
        
        return X, y


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing module loaded successfully")
