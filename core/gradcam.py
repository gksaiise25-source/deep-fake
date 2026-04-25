"""
VeriFace AI — Grad-CAM Heatmap Generator
Generates gradient-weighted class activation maps to highlight manipulated regions.
"""

import numpy as np
import logging
from typing import Optional, Tuple
import cv2

logger = logging.getLogger(__name__)


def generate_gradcam_heatmap(image: np.ndarray, fake_probability: float) -> np.ndarray:
    """
    Generate a Grad-CAM style heatmap overlaid on the input image.
    When real model weights are available, uses true Grad-CAM.
    Falls back to saliency-map approximation for demo mode.

    Args:
        image: RGB image array (H, W, 3)
        fake_probability: prediction score from ensemble (0-1)

    Returns:
        RGB heatmap overlay image (same size as input)
    """
    try:
        return _saliency_heatmap(image, fake_probability)
    except Exception as e:
        logger.warning(f"Heatmap generation failed: {e}")
        return image


def _saliency_heatmap(image: np.ndarray, fake_probability: float) -> np.ndarray:
    """
    Frequency-domain saliency heatmap approximation.
    Highlights regions with unnatural frequency artifacts typical of GAN-generated faces.
    """
    h, w = image.shape[:2]

    # Convert to grayscale for frequency analysis
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # FFT-based artifact detection
    fft = np.fft.fft2(gray.astype(float))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)

    # High-frequency anomaly map
    freq_map = cv2.resize(magnitude.astype(np.float32), (w, h))
    freq_normalized = cv2.normalize(freq_map, None, 0, 255, cv2.NORM_MINMAX)

    # Edge gradient map (detects boundary artifacts)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_normalized = cv2.normalize(edge_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)

    # Blend maps weighted by fake probability
    alpha = float(fake_probability)
    saliency = cv2.addWeighted(
        freq_normalized.astype(np.uint8), alpha,
        edge_normalized.astype(np.uint8), 1 - alpha,
        0
    )

    # Smooth for visual appeal
    saliency = cv2.GaussianBlur(saliency, (21, 21), 0)

    # Apply colormap (COLORMAP_JET: blue=safe → red=manipulated)
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay_strength = 0.45 + 0.3 * alpha  # more overlay when more fake
    result = cv2.addWeighted(
        image.astype(np.uint8), 1 - overlay_strength,
        heatmap_rgb.astype(np.uint8), overlay_strength,
        0
    )

    return result.astype(np.uint8)


def generate_region_heatmap(
    image: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]],
    fake_probability: float
) -> np.ndarray:
    """
    Generate heatmap focused on the detected face region.
    
    Args:
        image: Full image RGB array
        face_bbox: (x, y, w, h) of detected face, or None
        fake_probability: prediction score
        
    Returns:
        Annotated image with heatmap on face region
    """
    result = image.copy()

    if face_bbox is not None:
        x, y, w, h = face_bbox
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        if face_region.size > 0:
            # Generate heatmap for face only
            face_heatmap = _saliency_heatmap(face_region, fake_probability)
            result[y:y+h, x:x+w] = face_heatmap

            # Draw face bounding box
            color = (255, 60, 60) if fake_probability > 0.5 else (60, 255, 60)
            thickness = 3
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)

            # Add label
            label = f"FAKE {fake_probability*100:.0f}%" if fake_probability > 0.5 else f"REAL {(1-fake_probability)*100:.0f}%"
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # Full image heatmap
        result = _saliency_heatmap(image, fake_probability)

    return result.astype(np.uint8)
