"""
VeriFace AI — Metadata Tampering Checker (Enhanced for AI-Generated Images)

Key AI-image indicators:
  - PNG format with no camera EXIF (ChatGPT/DALL-E always outputs PNG)
  - No camera make/model
  - No DateTimeOriginal
  - Software signatures from AI tools
  - Square/power-of-2 dimensions (GAN output sizes)
"""

import logging
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Known AI/GAN software keywords
AI_SOFTWARE_SIGNATURES = [
    "stable diffusion", "midjourney", "dall-e", "dall·e", "openai",
    "deepfakes", "faceswap", "deepfacelab", "simswap", "insightface",
    "roop", "avatarify", "face2face", "neural", "gan", "generated",
    "synthetic", "artificial", "adobe firefly", "imagen", "gemini",
    "comfyui", "automatic1111", "invokeai", "diffusion",
]


def analyze_metadata(image_path: str) -> Dict:
    """
    Comprehensive metadata analysis.
    Returns a dict including 'tampering_score' (0-100) and 'ai_generated_score' (0-1).
    """
    path = Path(image_path)
    result = {
        'file_name': path.name,
        'file_size_kb': round(path.stat().st_size / 1024, 1) if path.exists() else 0,
        'extension': path.suffix.lower(),
        'exif_present': False,
        'tampering_signals': [],
        'tampering_score': 0,
        'software_detected': None,
        'gan_signature_found': False,
        'gps_data': None,
        'creation_date': None,
        'camera_info': None,
        'raw_fields': {},
        'ai_generated_score': 0.0,  # 0=real, 1=AI-generated
        'tampering_verdict': 'LOW RISK',
    }

    if not PIL_AVAILABLE:
        result['error'] = 'PIL not available'
        return result

    signals = []
    score = 0
    ai_score = 0.0

    try:
        img = PILImage.open(image_path)
        result['image_size'] = f"{img.width}x{img.height}"
        result['image_mode'] = img.mode
        ext = path.suffix.lower()

        # ── Signal 1: PNG format = strong AI indicator ─────────────────────
        # Real photos from cameras are always JPEG.
        # AI tools (ChatGPT, DALL-E, Midjourney, Stable Diffusion) output PNG.
        if ext in ['.png', '.webp']:
            signals.append(f"⚠️ {ext.upper()} format — real camera photos are JPEG; AI tools output PNG")
            score += 40
            ai_score += 0.40

        # ── Signal 2: No EXIF at all ────────────────────────────────────────
        exif_data = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    exif_data[tag] = value
        except Exception:
            pass

        # For PNG, try info dict
        if not exif_data and hasattr(img, 'info'):
            png_info = img.info or {}
            if 'exif' in png_info:
                signals.append("PNG has embedded EXIF — unusual for AI images")
            elif ext == '.png':
                signals.append("⚠️ PNG with no EXIF — AI-generated images have no camera metadata")
                score += 25
                ai_score += 0.25

        if not exif_data:
            result['exif_present'] = False
            if ext in ['.jpg', '.jpeg']:
                signals.append("⚠️ JPEG with no EXIF — very unusual for real camera photos")
                score += 30
                ai_score += 0.30
        else:
            result['exif_present'] = True

            # ── Signal 3: Camera make/model ─────────────────────────────────
            make = str(exif_data.get('Make', '')).strip()
            model = str(exif_data.get('Model', '')).strip()
            result['camera_info'] = f"{make} {model}".strip() or None

            if not make and not model:
                signals.append("⚠️ No camera make/model — all real camera photos have this")
                score += 25
                ai_score += 0.20
            else:
                ai_score -= 0.15  # Strong real indicator

            # ── Signal 4: AI software in EXIF ──────────────────────────────
            software = str(exif_data.get('Software', '')).lower()
            result['software_detected'] = exif_data.get('Software')
            for sig in AI_SOFTWARE_SIGNATURES:
                if sig in software:
                    signals.append(f"🚨 AI software signature: '{exif_data.get('Software', '')}'")
                    result['gan_signature_found'] = True
                    score += 50
                    ai_score += 0.50
                    break

            # ── Signal 5: Timestamps ────────────────────────────────────────
            date_original = exif_data.get('DateTimeOriginal')
            date_modified = exif_data.get('DateTime')
            result['creation_date'] = str(date_original) if date_original else None

            if not date_original:
                signals.append("⚠️ No capture timestamp — unusual for camera photos")
                score += 15
                ai_score += 0.10
            elif date_original and date_modified and str(date_original) != str(date_modified):
                signals.append("Timestamps differ — possible post-editing")
                score += 10

            # GPS
            result['gps_data'] = bool(exif_data.get('GPSInfo'))

            # Safe fields for display
            safe_tags = ['Make', 'Model', 'Software', 'DateTime', 'DateTimeOriginal',
                         'Flash', 'FocalLength', 'ISOSpeedRatings', 'ExposureTime']
            result['raw_fields'] = {t: str(exif_data[t]) for t in safe_tags if t in exif_data}

        # ── Signal 6: AI-typical dimensions ────────────────────────────────
        # AI image generators default to square/power-of-2 sizes
        ai_sizes = {256, 512, 768, 1024, 1152, 1280, 1344, 1536, 2048}
        if img.width == img.height and img.width in ai_sizes:
            signals.append(f"⚠️ Square {img.width}×{img.height} — common GAN/diffusion output size")
            score += 20
            ai_score += 0.15
        elif img.width in ai_sizes or img.height in ai_sizes:
            signals.append(f"Dimension matches common AI output size ({img.width}×{img.height})")
            score += 8

        # ── Signal 7: File size anomaly ─────────────────────────────────────
        # AI PNG images tend to be very large (lossless), real JPEG photos are compact
        size_kb = result['file_size_kb']
        if ext == '.png' and size_kb > 1500:
            signals.append(f"⚠️ Large PNG ({size_kb:.0f} KB) — AI tools save uncompressed PNG")
            score += 10
            ai_score += 0.05

    except Exception as e:
        logger.warning(f"Metadata analysis error: {e}")
        result['error'] = str(e)

    result['tampering_signals'] = signals
    result['tampering_score'] = min(score, 100)
    result['ai_generated_score'] = float(min(max(ai_score, 0.0), 1.0))
    result['tampering_verdict'] = (
        "HIGH RISK — Likely AI Generated" if score >= 55 else
        "MODERATE RISK" if score >= 30 else
        "LOW RISK"
    )

    return result
