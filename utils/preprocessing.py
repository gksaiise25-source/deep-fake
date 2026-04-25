"""
VeriFace AI — Image & Video Preprocessing Utilities
"""

import cv2
import numpy as np
import logging
import tempfile
import os
from typing import List, Optional, Tuple
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def load_image_rgb(source) -> Optional[np.ndarray]:
    """
    Load image from file path, bytes, or PIL Image into RGB numpy array.
    """
    try:
        if isinstance(source, np.ndarray):
            return source if source.ndim == 3 else cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        if isinstance(source, PILImage.Image):
            return np.array(source.convert("RGB"))
        if isinstance(source, (str, os.PathLike)):
            img = cv2.imread(str(source))
            if img is None:
                raise ValueError(f"Cannot read image: {source}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(source, bytes):
            arr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Unsupported source type: {type(source)}")
    except Exception as e:
        logger.error(f"load_image_rgb failed: {e}")
        return None


def preprocess_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True
) -> np.ndarray:
    """Resize and normalize image for model inference."""
    resized = cv2.resize(image.astype(np.uint8), target_size, interpolation=cv2.INTER_LANCZOS4)
    if normalize:
        return resized.astype(np.float32) / 255.0
    return resized.astype(np.float32)


def extract_video_frames(
    video_path: str,
    max_frames: int = 60,
    return_rgb: bool = True
) -> List[np.ndarray]:
    """
    Extract evenly-spaced frames from a video file.
    
    Returns:
        List of RGB frame arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // max_frames)
    frames = []
    idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            if return_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def save_temp_file(data: bytes, suffix: str) -> str:
    """Save bytes to a temporary file, return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        return f.name


def image_to_bytes(image: np.ndarray, format: str = "JPEG") -> bytes:
    """Convert RGB numpy array to image bytes."""
    pil = PILImage.fromarray(image.astype(np.uint8))
    buf = __import__('io').BytesIO()
    pil.save(buf, format=format, quality=95)
    return buf.getvalue()
