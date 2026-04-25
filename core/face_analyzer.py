"""
VeriFace AI — Face Analyzer
Detects faces and analyzes facial inconsistencies indicating deepfake manipulation.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MTCNN_AVAILABLE = False
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    logger.warning("MTCNN not available — using OpenCV fallback")

_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_opencv_face_cascade = cv2.CascadeClassifier(_cascade_path)


class FaceAnalyzer:
    def __init__(self):
        self.detector = None
        if MTCNN_AVAILABLE:
            try:
                self.detector = MTCNN()
            except Exception as e:
                logger.warning(f"MTCNN init failed: {e}")

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        if self.detector:
            return self._detect_mtcnn(image)
        return self._detect_opencv(image)

    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        try:
            results = self.detector.detect_faces(image)
            return [{
                'bbox': (max(0, r['box'][0]), max(0, r['box'][1]), r['box'][2], r['box'][3]),
                'confidence': float(r['confidence']),
                'landmarks': r.get('keypoints', {}),
                'method': 'mtcnn'
            } for r in results]
        except Exception:
            return self._detect_opencv(image)

    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detected = _opencv_face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(detected) == 0:
            return []
        return [{'bbox': (int(x), int(y), int(w), int(h)), 'confidence': 0.85, 'landmarks': {}, 'method': 'opencv'}
                for (x, y, w, h) in detected]

    def analyze_inconsistencies(self, image: np.ndarray, faces: List[Dict]) -> Dict:
        if not faces:
            return self._empty_report()

        face = faces[0]
        x, y, w, h = face['bbox']
        face_region = image[y:y+h, x:x+w]
        if face_region.size == 0:
            return self._empty_report()

        scores = {
            'skin_texture': self._skin_texture(face_region),
            'boundary_artifacts': self._boundary_artifacts(image, face),
            'lighting_inconsistency': self._lighting(face_region),
            'eye_anomaly': self._eye_anomaly(face_region),
            'color_inconsistency': self._color_stats(face_region),
            'frequency_artifacts': self._frequency_artifacts(face_region),
        }
        weights = [0.25, 0.20, 0.15, 0.20, 0.10, 0.10]
        overall = float(np.dot(list(scores.values()), weights))

        return {
            'face_detected': True,
            'face_count': len(faces),
            'primary_face_bbox': face['bbox'],
            'detection_confidence': face['confidence'],
            'detection_method': face['method'],
            'inconsistency_scores': scores,
            'overall_inconsistency': round(overall, 2),
            'landmarks_available': bool(face.get('landmarks'))
        }

    def _skin_texture(self, face: np.ndarray) -> float:
        gray = cv2.cvtColor(face.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(np.clip(1.0 / (1.0 + var / 500) * 100, 0, 100))

    def _boundary_artifacts(self, image: np.ndarray, face: Dict) -> float:
        x, y, w, h = face['bbox']
        m = 10
        inner = image[y+m:y+h-m, x+m:x+w-m]
        if inner.size == 0:
            return 20.0
        inner_std = np.std(inner.astype(float))
        outer = []
        if y > m: outer.append(image[max(0,y-m):y, x:x+w])
        if y+h+m < image.shape[0]: outer.append(image[y+h:y+h+m, x:x+w])
        if not outer:
            return 20.0
        outer_std = np.std(np.concatenate(outer).astype(float))
        disc = abs(inner_std - outer_std) / (max(inner_std, outer_std) + 1e-6)
        return float(np.clip(disc * 100, 0, 100))

    def _lighting(self, face: np.ndarray) -> float:
        gray = cv2.cvtColor(face.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        la, ra = np.mean(gray[:, :w//2]), np.mean(gray[:, w//2:])
        asym = abs(la - ra) / (max(la, ra) + 1e-6)
        return float(np.clip(asym * 70, 0, 100))

    def _eye_anomaly(self, face: np.ndarray) -> float:
        h = face.shape[0]
        eye_region = face[:int(h * 0.45), :]
        if eye_region.size == 0:
            return 30.0
        gray = cv2.cvtColor(eye_region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(np.clip(max(0, 100 - sharpness / 10), 0, 100))

    def _color_stats(self, face: np.ndarray) -> float:
        stds = [np.std(face[:,:,c].astype(float)) for c in range(3)]
        return float(np.clip(np.var(stds) / 50 * 60, 0, 100))

    def _frequency_artifacts(self, face: np.ndarray) -> float:
        gray = cv2.cvtColor(face.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        fft = np.fft.fftshift(np.fft.fft2(gray.astype(float)))
        mag = np.abs(fft)
        h, w = mag.shape
        center = mag[h//4:3*h//4, w//4:3*w//4].mean()
        mask = np.ones_like(mag, bool)
        mask[h//4:3*h//4, w//4:3*w//4] = False
        high = mag[mask].mean() if mask.any() else 0
        ratio = high / (center + 1e-6)
        return float(np.clip(ratio * 40, 0, 100))

    def _empty_report(self) -> Dict:
        return {
            'face_detected': False, 'face_count': 0, 'primary_face_bbox': None,
            'detection_confidence': 0.0, 'detection_method': 'none',
            'inconsistency_scores': {k: 0 for k in [
                'skin_texture', 'boundary_artifacts', 'lighting_inconsistency',
                'eye_anomaly', 'color_inconsistency', 'frequency_artifacts'
            ]},
            'overall_inconsistency': 0.0, 'landmarks_available': False
        }


_analyzer: Optional[FaceAnalyzer] = None

def get_face_analyzer() -> FaceAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = FaceAnalyzer()
    return _analyzer
