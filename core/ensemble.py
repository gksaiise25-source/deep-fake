"""
VeriFace AI — Forensic Deepfake Detection Engine v2

Detection covers two categories:
  A) Basic GAN / deepfake swap: low Laplacian, smooth textures, GAN noise
  B) High-quality AI generation (ChatGPT/DALL-E/Midjourney): photorealistic but
     detectable via metadata (PNG, no EXIF), face skin hyper-smoothness, and
     frequency fingerprints

Signal architecture:
  1. Metadata forensics (PNG format, no EXIF, no camera) — weight 0.35
  2. Face-region skin texture analysis (over-smooth = AI)  — weight 0.25
  3. Laplacian sharpness + local texture variance           — weight 0.20
  4. FFT spectral slope (1/f law)                          — weight 0.12
  5. Noise floor naturalness                               — weight 0.08
"""

import os
import logging
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import time

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T
    from PIL import Image as PILImage
    TORCH_AVAILABLE = True
except ImportError:
    pass


# ─── Signal 1: Metadata-based AI detection ───────────────────────────────────
def sig_metadata(metadata_result: Optional[Dict]) -> float:
    """
    Returns fake score based on metadata analysis.
    PNG + no EXIF = very strong AI indicator.
    """
    if metadata_result is None:
        return 0.30  # Unknown

    ai_score = metadata_result.get('ai_generated_score', None)
    if ai_score is not None:
        return float(np.clip(ai_score, 0, 1))

    # Fallback: compute from tampering_score
    ts = metadata_result.get('tampering_score', 0)
    return float(np.clip(ts / 100.0, 0, 1))


# ─── Signal 2: Face-region skin texture analysis ─────────────────────────────
def sig_face_skin_texture(image: np.ndarray) -> Tuple[float, bool]:
    """
    Analyzes face regions for AI-characteristic hyper-smooth skin.
    
    AI portrait generators (DALL-E, MJ, SD) create faces with:
    - Very uniform, smooth skin texture in the face region
    - Laplacian variance in face region << background
    - Local std of face pixels very low
    
    Returns: (fake_score, face_found)
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Detect faces using Haar cascade (built-in OpenCV, no download needed)
    cascade_paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    face_cascade = None
    for cp in cascade_paths:
        if os.path.exists(cp):
            face_cascade = cv2.CascadeClassifier(cp)
            break

    if face_cascade is None or face_cascade.empty():
        # No face detection available — use center region
        face_rects = [(w//4, h//4, w//2, h//2)]
        face_found = False
    else:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            # Try more permissive
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
            )

        if len(faces) == 0:
            face_rects = [(w//4, h//4, w//2, h//2)]
            face_found = False
        else:
            face_rects = faces.tolist()
            face_found = True

    face_scores = []
    for (fx, fy, fw, fh) in face_rects[:3]:
        # Extract face region (use inner 60% to avoid hair/background)
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.20)
        x1 = max(0, fx + pad_x)
        y1 = max(0, fy + pad_y)
        x2 = min(w, fx + fw - pad_x)
        y2 = min(h, fy + fh - pad_y)

        face_gray = gray[y1:y2, x1:x2]
        face_color = image[y1:y2, x1:x2]

        if face_gray.size < 400:
            continue

        # ── Skin texture Laplacian in face region ──────────────────────────
        face_lap = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())

        # Real portrait: face Laplacian 40-400 (natural skin texture, pores)
        # AI portrait: face Laplacian 5-35 (over-smoothed, waxy skin)
        if face_lap < 15:
            lap_score = 0.85   # Extremely smooth = AI
        elif face_lap < 30:
            lap_score = 0.65   # Very smooth = suspicious
        elif face_lap < 60:
            lap_score = 0.40   # Moderately smooth = uncertain
        elif face_lap < 150:
            lap_score = 0.15   # Natural texture = real
        else:
            lap_score = 0.08   # Very detailed = definitely real

        # ── Local pixel uniformity in face region ──────────────────────────
        # Compute std of 8x8 blocks within face
        block_stds = []
        fh_r, fw_r = face_gray.shape
        for i in range(0, fh_r - 8, 8):
            for j in range(0, fw_r - 8, 8):
                block_stds.append(float(np.std(face_gray[i:i+8, j:j+8].astype(float))))

        if block_stds:
            mean_block_std = float(np.mean(block_stds))
            # AI: mean block std < 8 (super uniform skin)
            # Real: mean block std > 12 (varied pores, shadows)
            if mean_block_std < 6:
                uni_score = 0.80
            elif mean_block_std < 10:
                uni_score = 0.55
            elif mean_block_std < 18:
                uni_score = 0.25
            else:
                uni_score = 0.10
        else:
            uni_score = 0.35

        face_score = 0.55 * lap_score + 0.45 * uni_score
        face_scores.append(face_score)

    if not face_scores:
        return 0.35, False

    return float(np.mean(face_scores)), face_found


# ─── Signal 3: Laplacian sharpness (whole image) ─────────────────────────────
def sig_laplacian(gray: np.ndarray) -> float:
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var >= 120:
        return 0.05
    elif lap_var >= 60:
        return 0.15
    elif lap_var >= 25:
        return 0.40
    elif lap_var >= 8:
        return 0.70
    else:
        return 0.88


# ─── Signal 4: FFT spectral slope ────────────────────────────────────────────
def sig_fft_slope(gray: np.ndarray) -> float:
    h, w = gray.shape
    fft = np.fft.fft2(gray.astype(float) / 255.0)
    power = np.abs(np.fft.fftshift(fft)) ** 2
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    radial_power = []
    for r in range(2, max_r, max(1, max_r // 20)):
        y_g, x_g = np.ogrid[:h, :w]
        ring = (np.sqrt((y_g - cy)**2 + (x_g - cx)**2).round().astype(int) == r)
        if ring.any():
            radial_power.append((r, float(power[ring].mean())))

    if len(radial_power) < 5:
        return 0.5

    radii = np.log([rp[0] for rp in radial_power])
    powers = np.log([max(rp[1], 1e-12) for rp in radial_power])
    try:
        slope = float(np.polyfit(radii, powers, 1)[0])
    except Exception:
        return 0.5

    if -2.8 <= slope <= -1.2:
        return 0.10
    elif -3.5 <= slope <= -0.8:
        return 0.35
    else:
        return 0.65


# ─── Signal 5: Noise floor ────────────────────────────────────────────────────
def sig_noise(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0).astype(float)
    noise = gray.astype(float) - blur
    h, w = noise.shape

    block_sigmas = []
    for i in range(0, h - 32, 16):
        for j in range(0, w - 32, 16):
            block_sigmas.append(float(np.std(noise[i:i+32, j:j+32])))

    if len(block_sigmas) < 4:
        return 0.4

    mean_sigma = float(np.mean(block_sigmas))
    cv_sigma = float(np.std(block_sigmas)) / (mean_sigma + 1e-6)

    if 1.5 <= mean_sigma <= 10 and 0.2 <= cv_sigma <= 2.0:
        return 0.10
    elif mean_sigma < 0.8:
        return 0.75
    elif mean_sigma > 15:
        return 0.60
    else:
        return 0.40


# ─── Main Ensemble ────────────────────────────────────────────────────────────
class EnsembleDetector:
    """
    Multi-signal forensic ensemble.

    Weights:
      metadata_signal:     0.35  (PNG, no EXIF, no camera → AI generated)
      face_skin:           0.25  (AI faces are hyper-smooth)
      image_laplacian:     0.20  (whole image sharpness)
      fft_slope:           0.12  (spectral fingerprint)
      noise_floor:         0.08  (sensor noise pattern)

    Threshold: 0.50
    """

    WEIGHTS = {
        'metadata':  0.35,
        'face_skin': 0.25,
        'laplacian': 0.20,
        'fft':       0.12,
        'noise':     0.08,
    }

    def __init__(self, resnet18_path: str = "models/deepfake_resnet18.pth"):
        self.is_demo = False
        logger.info("✅ VeriFace AI Forensic Ensemble v2 ready")

    def analyze(self, image: np.ndarray, metadata_result: Optional[Dict] = None) -> Dict:
        start = time.time()
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Compute all signals
        meta_score = sig_metadata(metadata_result)
        face_score, face_found = sig_face_skin_texture(image)
        lap_score = sig_laplacian(gray)
        fft_score = sig_fft_slope(gray)
        noise_score = sig_noise(gray)

        scores = {
            'metadata':  meta_score,
            'face_skin': face_score,
            'laplacian': lap_score,
            'fft':       fft_score,
            'noise':     noise_score,
        }

        ensemble_score = sum(scores[k] * self.WEIGHTS[k] for k in scores)

        # If metadata is very strong (PNG + no EXIF → score ≥ 0.60), boost
        if meta_score >= 0.60:
            ensemble_score = max(ensemble_score, 0.62)

        ensemble_score = float(np.clip(ensemble_score, 0.01, 0.99))
        is_fake = ensemble_score > 0.50
        confidence = float(max(ensemble_score, 1 - ensemble_score) * 100)
        elapsed = time.time() - start

        return {
            "is_fake": bool(is_fake),
            "label": "DEEPFAKE" if is_fake else "AUTHENTIC",
            "fake_probability": ensemble_score,
            "confidence": confidence,
            "deepfake_percentage": ensemble_score * 100,
            "per_model_scores": {
                "Metadata Forensics":    round(meta_score * 100, 1),
                "Face Skin Texture":     round(face_score * 100, 1),
                "Image Sharpness":       round(lap_score * 100, 1),
                "Spectral Analysis":     round(fft_score * 100, 1),
                "Noise Pattern":         round(noise_score * 100, 1),
            },
            "signal_breakdown": scores,
            "face_detected": face_found,
            "neural_raw_score": None,
            "inference_time_ms": round(elapsed * 1000, 1),
            "is_demo_mode": False,
        }


# ─── Singleton ────────────────────────────────────────────────────────────────
_detector: Optional[EnsembleDetector] = None


def get_detector() -> EnsembleDetector:
    global _detector
    if _detector is None:
        path = os.getenv("RESNET18_MODEL_PATH", "models/deepfake_resnet18.pth")
        _detector = EnsembleDetector(resnet18_path=path)
    return _detector
