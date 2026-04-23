"""
Data Augmentation Script for Deepfake Detection

Augments training images to expand the dataset and improve model robustness.
Generates multiple variations of each image through transformations.

Run with:
    python augment_data.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmenter:
    """Apply various augmentations to images"""
    
    def __init__(self, output_multiplier=4):
        """
        Args:
            output_multiplier: How many augmented versions per image (3-5 recommended)
        """
        self.output_multiplier = output_multiplier
    
    def augment_image(self, image):
        """Apply random augmentations to image"""
        augmented_images = [image]  # Include original
        
        for _ in range(self.output_multiplier - 1):
            aug_img = image.copy()
            
            # Random rotation (-15 to +15 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = aug_img.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                aug_img = cv2.warpAffine(aug_img, matrix, (w, h))
            
            # Random brightness adjustment
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)
            
            # Random horizontal flip
            if np.random.rand() > 0.5:
                aug_img = cv2.flip(aug_img, 1)
            
            # Random Gaussian blur
            if np.random.rand() > 0.5:
                kernel_size = np.random.choice([3, 5, 7])
                aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
            
            # Random noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 5, aug_img.shape)
                aug_img = np.clip(aug_img + noise, 0, 255).astype(np.uint8)
            
            # Random vertical shift
            if np.random.rand() > 0.5:
                shift = np.random.randint(-10, 10)
                matrix = np.float32([[1, 0, 0], [0, 1, shift]])
                aug_img = cv2.warpAffine(aug_img, matrix, (aug_img.shape[1], aug_img.shape[0]))
            
            # Random horizontal shift
            if np.random.rand() > 0.5:
                shift = np.random.randint(-10, 10)
                matrix = np.float32([[1, 0, shift], [0, 1, 0]])
                aug_img = cv2.warpAffine(aug_img, matrix, (aug_img.shape[1], aug_img.shape[0]))
            
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def augment_directory(self, input_dir, output_dir):
        """Augment all images in directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        logger.info(f"Found {len(image_files)} images to augment")
        
        for filename in tqdm(image_files, desc=f"Augmenting {os.path.basename(input_dir)}"):
            filepath = os.path.join(input_dir, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                logger.warning(f"Could not read {filename}")
                continue
            
            # Get augmented versions
            augmented = self.augment_image(image)
            
            # Save original + augmented
            name, ext = os.path.splitext(filename)
            
            # Save original
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
            
            # Save augmented versions
            for i, aug_img in enumerate(augmented[1:], 1):
                aug_filename = f"{name}_aug{i}{ext}"
                aug_path = os.path.join(output_dir, aug_filename)
                cv2.imwrite(aug_path, aug_img)
        
        logger.info(f"✅ Augmented images saved to {output_dir}")
        count = len(os.listdir(output_dir))
        logger.info(f"Total images: {count}")


def main():
    """Main augmentation pipeline"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # Original paths
    train_real = os.path.join(data_dir, 'train_real')
    train_fake = os.path.join(data_dir, 'train_fake')
    
    # Augmented output paths
    train_real_aug = os.path.join(data_dir, 'train_real_augmented')
    train_fake_aug = os.path.join(data_dir, 'train_fake_augmented')
    
    if not os.path.exists(train_real) or not os.path.exists(train_fake):
        print("❌ Training data not found!")
        print(f"Expected: {train_real} and {train_fake}")
        return
    
    # Remove old augmented data (skip if permission denied)
    for aug_dir in [train_real_aug, train_fake_aug]:
        if os.path.exists(aug_dir):
            try:
                shutil.rmtree(aug_dir)
                logger.info(f"Removed old augmented data: {aug_dir}")
            except PermissionError:
                logger.warning(f"⚠️  Permission denied removing {aug_dir} - will overwrite instead")
            except Exception as e:
                logger.warning(f"⚠️  Could not remove {aug_dir}: {e}")
    
    # Augment data
    augmenter = ImageAugmenter(output_multiplier=4)
    
    print("\n" + "="*60)
    print("🔄 DATA AUGMENTATION")
    print("="*60 + "\n")
    
    print("Augmenting REAL images...")
    augmenter.augment_directory(train_real, train_real_aug)
    
    print("\nAugmenting FAKE images...")
    augmenter.augment_directory(train_fake, train_fake_aug)
    
    # Summary
    real_count = len(os.listdir(train_real_aug))
    fake_count = len(os.listdir(train_fake_aug))
    
    print("\n" + "="*60)
    print("✅ AUGMENTATION COMPLETE")
    print("="*60)
    print(f"📊 REAL images: {real_count} (from {len(os.listdir(train_real))})")
    print(f"📊 FAKE images: {fake_count} (from {len(os.listdir(train_fake))})")
    print(f"📊 Total dataset: {real_count + fake_count} images")
    print(f"📈 Data expansion: {(real_count + fake_count) / (len(os.listdir(train_real)) + len(os.listdir(train_fake)))}x")
    print("\nNext: Train model with augmented data using train_advanced.py")


if __name__ == "__main__":
    main()
