"""
Advanced Feature Extraction for Deepfake Detection

Implements sophisticated feature extraction techniques:
- Frequency domain analysis (FFT, DCT)
- Compression artifacts detection
- Inconsistency detection
- Eye region analysis
- Facial boundary analysis
"""

import numpy as np
import cv2
from scipy import signal
from scipy.fftpack import dct
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureExtractor:
    """Extract advanced features for deepfake detection"""
    
    def __init__(self):
        self.target_size = (256, 256)
    
    def extract_features(self, image):
        """Extract all advanced features from image"""
        if image is None:
            return np.zeros(168)
        
        # Ensure image is in range [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        features = []
        
        # 1. Frequency domain features (DCT)
        dct_features = self._extract_dct_features(image)
        features.extend(dct_features)
        
        # 2. Compression artifacts
        compression_features = self._extract_compression_artifacts(image)
        features.extend(compression_features)
        
        # 3. Noise inconsistency
        noise_features = self._extract_noise_features(image)
        features.extend(noise_features)
        
        # 4. Color space anomalies
        color_features = self._extract_color_anomalies(image)
        features.extend(color_features)
        
        # 5. Frequency imbalance
        freq_features = self._extract_frequency_features(image)
        features.extend(freq_features)
        
        # 6. Blurring detection
        blur_features = self._extract_blur_features(image)
        features.extend(blur_features)
        
        # 7. Edge inconsistency
        edge_features = self._extract_edge_features(image)
        features.extend(edge_features)
        
        # 8. Lighting consistency
        lighting_features = self._extract_lighting_features(image)
        features.extend(lighting_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_dct_features(self, image):
        """
        DCT-based frequency features
        AI-generated images have different DCT coefficient distributions
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Compute 2D DCT
        dct_matrix = dct(dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_abs = np.abs(dct_matrix)
        
        features = []
        # Energy in different frequency bands
        h, w = dct_abs.shape
        
        # Low frequency (upper left)
        low_freq = dct_abs[:h//4, :w//4].mean()
        features.append(low_freq)
        
        # Mid frequency
        mid_freq = dct_abs[h//4:h//2, w//4:w//2].mean()
        features.append(mid_freq)
        
        # High frequency
        high_freq = dct_abs[h//2:, w//2:].mean()
        features.append(high_freq)
        
        # Frequency skewness (ratio of low to high)
        features.append(low_freq / (high_freq + 1e-5))
        
        # DCT coefficient variance
        features.append(dct_abs.var())
        
        return features  # 5 features
    
    def _extract_compression_artifacts(self, image):
        """
        Detect compression artifacts (block patterns)
        Real images: complex artifacts; AI-generated: smooth or structured artifacts
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # Block-level variance (JPEG artifacts)
        block_size = 8
        h, w = gray.shape
        block_vars = []
        
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_vars.append(block.var())
        
        if block_vars:
            features.append(np.mean(block_vars))
            features.append(np.std(block_vars))
        else:
            features.extend([0, 0])
        
        # Edge strength within blocks vs across blocks
        edges = cv2.Canny(gray, 50, 150)
        
        internal_edges = 0
        boundary_edges = 0
        
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block = edges[i:i+block_size, j:j+block_size]
                internal_edges += block[1:-1, 1:-1].sum()
                boundary_edges += (block[0, :].sum() + block[-1, :].sum() + 
                                 block[:, 0].sum() + block[:, -1].sum())
        
        features.append(internal_edges / (boundary_edges + 1e-5))
        
        return features  # 3 features
    
    def _extract_noise_features(self, image):
        """
        Detect noise inconsistency patterns
        AI-generated images have inconsistent or structured noise
        """
        gray = (image[:,:,0] * 0.299 + image[:,:,1] * 0.587 + image[:,:,2] * 0.114)
        gray = (gray * 255).astype(np.uint8)
        
        features = []
        
        # Laplacian variance (sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(laplacian.var())
        
        # Difference of Gaussian
        blur1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
        blur2 = cv2.GaussianBlur(gray, (9, 9), 2.0)
        dog = blur1.astype(np.float32) - blur2.astype(np.float32)
        features.append(dog.var())
        
        # Local binary patterns (simplified)
        lbp_hist = self._extract_lbp_histogram(gray)
        features.extend(lbp_hist[:8])
        
        return features  # 10 features
    
    def _extract_lbp_histogram(self, image, num_bins=8):
        """Extract Local Binary Pattern histogram"""
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                binary = 0
                for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), 
                                               (1,1), (1,0), (1,-1), (0,-1)]):
                    if image[i+di, j+dj] >= center:
                        binary |= (1 << k)
                lbp[i-1, j-1] = binary
        
        hist, _ = np.histogram(lbp, bins=num_bins, range=(0, 256))
        return hist / (hist.sum() + 1e-5)
    
    def _extract_color_anomalies(self, image):
        """
        Detect color space anomalies
        AI-generated images often have subtle color inaccuracies
        """
        features = []
        
        # RGB correlation (should be high in natural images)
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        features.append(np.corrcoef(r.flatten(), g.flatten())[0, 1])
        features.append(np.corrcoef(g.flatten(), b.flatten())[0, 1])
        features.append(np.corrcoef(r.flatten(), b.flatten())[0, 1])
        
        # Convert to HSV and check saturation
        img_hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        saturation = img_hsv[:,:,1] / 255.0
        features.append(saturation.mean())
        features.append(saturation.std())
        
        return features  # 5 features
    
    def _extract_frequency_features(self, image):
        """
        Extract FFT-based frequency features
        AI-generated images have different frequency distributions
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Compute FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        features = []
        
        # Log magnitude average (spectral energy)
        log_mag = np.log1p(magnitude)
        features.append(log_mag.mean())
        features.append(log_mag.std())
        
        # Radial average (1D frequency profile)
        h, w = magnitude.shape
        cx, cy = h // 2, w // 2
        max_r = min(cx, cy)
        
        radial_avg = []
        for r in range(0, max_r, max(1, max_r // 8)):
            mask = np.zeros_like(magnitude)
            cv2.circle(mask, (cy, cx), r, 1, 1)
            radial_avg.append(magnitude[mask > 0].mean())
        
        features.extend(radial_avg[:8])
        
        return features  # 10 features
    
    def _extract_blur_features(self, image):
        """
        Detect blurring/smoothness (common in AI-generated images)
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # Tenenhaus focus measure
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(laplacian.var())
        
        # Gaussian blur detection
        blur_kernels = [3, 5, 7]
        blur_scores = []
        gray_float = gray.astype(np.float32)
        
        for kernel_size in blur_kernels:
            blurred = cv2.GaussianBlur(gray_float, (kernel_size, kernel_size), 0)
            diff = np.abs(gray_float - blurred)
            blur_scores.append(diff.mean())
        
        features.extend(blur_scores)
        
        return features  # 4 features
    
    def _extract_edge_features(self, image):
        """
        Extract edge-based features
        AI images often have smoother edges or inconsistent edge patterns
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        
        features.append(sobel_mag.mean())
        features.append(sobel_mag.std())
        
        # Canny edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(edges.sum() / edges.size)
        
        # Edge direction histogram
        angles = np.arctan2(sobely, sobelx)
        hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        features.extend(hist / hist.sum())
        
        return features  # 12 features
    
    def _extract_lighting_features(self, image):
        """
        Detect lighting inconsistencies
        Common in AI-generated images and deepfakes
        """
        features = []
        
        # Illumination map (rough estimate)
        blur = cv2.GaussianBlur((image * 255).astype(np.uint8), (51, 51), 0)
        illum = blur.astype(np.float32) / 255.0
        
        # Statistics on illumination
        features.append(illum.mean())
        features.append(illum.std())
        
        # Shadow detection (dark regions)
        shadows = (illum < 0.3).sum() / illum.size
        features.append(shadows)
        
        # Highlight detection (bright regions)
        highlights = (illum > 0.7).sum() / illum.size
        features.append(highlights)
        
        return features  # 4 features
