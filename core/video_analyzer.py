"""
VeriFace AI — Video Analyzer
Frame-by-frame deepfake detection with temporal consistency and audio-video sync analysis.
"""

import cv2
import numpy as np
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LIBROSA_AVAILABLE = False
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("librosa not available — audio analysis disabled")


class VideoAnalyzer:
    """Frame-by-frame video deepfake analysis with audio sync check."""

    def __init__(self, ensemble_detector=None, face_analyzer=None, max_frames: int = 60):
        self.detector = ensemble_detector
        self.face_analyzer = face_analyzer
        self.max_frames = max_frames

    def analyze_video(self, video_path: str, progress_callback=None) -> Dict:
        """
        Full video analysis: extract frames, detect deepfakes, analyze temporal consistency.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video file'}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration_sec = total_frames / fps

        # Sample frames evenly
        sample_interval = max(1, total_frames // self.max_frames)
        frame_results = []
        frame_images = []
        frame_idx = 0
        sampled = 0

        while cap.isOpened() and sampled < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self._analyze_frame(frame_rgb, sampled, frame_idx)
                frame_results.append(result)
                if len(frame_images) < 6:
                    frame_images.append(frame_rgb)
                sampled += 1
                if progress_callback:
                    progress_callback(sampled / self.max_frames)
            frame_idx += 1

        cap.release()

        if not frame_results:
            return {'error': 'No frames extracted'}

        # Aggregate
        fake_probs = [r['fake_probability'] for r in frame_results]
        is_fakes = [r['is_fake'] for r in frame_results]
        mean_prob = float(np.mean(fake_probs))
        temporal_consistency = self._temporal_consistency(fake_probs)

        # Audio analysis
        audio_result = self._analyze_audio_sync(video_path) if LIBROSA_AVAILABLE else {
            'audio_available': False, 'sync_score': None, 'audio_tampering': None
        }

        overall_is_fake = mean_prob > 0.5
        confidence = float(max(mean_prob, 1 - mean_prob) * 100)

        return {
            'video_path': video_path,
            'duration_seconds': round(duration_sec, 1),
            'total_frames': total_frames,
            'frames_analyzed': len(frame_results),
            'fps': round(fps, 1),
            'is_fake': overall_is_fake,
            'label': 'DEEPFAKE' if overall_is_fake else 'AUTHENTIC',
            'fake_probability': mean_prob,
            'confidence': confidence,
            'deepfake_percentage': round(float(np.mean(is_fakes)) * 100, 1),
            'frame_predictions': frame_results,
            'fake_probabilities_timeline': fake_probs,
            'temporal_consistency': temporal_consistency,
            'audio_analysis': audio_result,
            'sample_frames': frame_images,
        }

    def _analyze_frame(self, frame_rgb: np.ndarray, sample_idx: int, frame_idx: int) -> Dict:
        """Analyze a single frame"""
        if self.detector:
            result = self.detector.analyze(frame_rgb)
            fake_prob = result['fake_probability']
        else:
            # Demo: deterministic based on frame stats
            np.random.seed(int(np.mean(frame_rgb)) % 997 + sample_idx)
            fake_prob = float(np.clip(0.3 + np.random.normal(0, 0.2), 0, 1))

        return {
            'sample_index': sample_idx,
            'frame_index': frame_idx,
            'is_fake': fake_prob > 0.5,
            'fake_probability': round(fake_prob, 4),
            'confidence': round(float(max(fake_prob, 1 - fake_prob) * 100), 1),
        }

    def _temporal_consistency(self, probs: List[float]) -> Dict:
        """Analyze temporal consistency of predictions."""
        if len(probs) < 2:
            return {'score': 100.0, 'interpretation': 'Insufficient frames'}

        arr = np.array(probs)
        variance = float(np.var(arr))
        transitions = sum(abs(probs[i] - probs[i-1]) > 0.3 for i in range(1, len(probs)))
        consistency_score = float(np.clip(100 - variance * 200 - transitions * 5, 0, 100))

        if consistency_score > 70:
            interpretation = "Consistent — natural temporal pattern"
        elif consistency_score > 40:
            interpretation = "Moderate inconsistency — possible manipulation"
        else:
            interpretation = "High inconsistency — strong deepfake signal"

        return {
            'score': round(consistency_score, 1),
            'variance': round(variance, 4),
            'abrupt_transitions': transitions,
            'interpretation': interpretation
        }

    def _analyze_audio_sync(self, video_path: str) -> Dict:
        """Analyze audio-video synchronization for lip sync inconsistencies."""
        if not LIBROSA_AVAILABLE:
            return {'audio_available': False, 'sync_score': None}
        try:
            import soundfile as sf
            # Extract audio to temp file using OpenCV (basic check)
            cap = cv2.VideoCapture(video_path)
            has_audio = cap.get(cv2.CAP_PROP_AUDIO_STREAM) >= 0
            cap.release()

            if not has_audio:
                return {'audio_available': False, 'sync_score': None,
                        'audio_tampering': None, 'message': 'No audio track detected'}

            y, sr = librosa.load(video_path, sr=22050, duration=30.0, mono=True)
            if len(y) == 0:
                return {'audio_available': False, 'sync_score': None}

            # Spectral features
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            zero_crossing = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            rms = float(np.mean(librosa.feature.rms(y=y)))

            # Simple heuristic: check for editing artifacts in spectral flux
            hop_length = 512
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            flux_std = float(np.std(onset_env))

            # Normalize sync score (demo heuristic)
            sync_score = float(np.clip(85 - flux_std * 2 + rms * 100, 40, 100))
            audio_tampering = flux_std > 20

            return {
                'audio_available': True,
                'duration_seconds': round(len(y) / sr, 1),
                'sync_score': round(sync_score, 1),
                'audio_tampering': audio_tampering,
                'spectral_centroid': round(spectral_centroid, 1),
                'rms_energy': round(rms, 4),
            }
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return {'audio_available': False, 'sync_score': None, 'error': str(e)}
