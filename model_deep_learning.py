"""
Deep Learning Model for Deepfake Detection using PyTorch

Uses ResNet18 pre-trained on ImageNet with transfer learning.
Provides superior performance compared to hand-crafted features.

Requires: torch, torchvision
Install with: pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import os
import cv2
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""
    
    def __init__(self, image_dir, label, transform=None):
        """
        Args:
            image_dir: Directory containing images
            label: 0 for real, 1 for fake
            transform: Image transformations
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.image_paths.append(os.path.join(image_dir, filename))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        
        if image is None:
            # If read fails, return black image
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # HWC to CHW
            # Normalize ImageNet mean/std
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
        
        return image, self.labels[idx]


class DeepfakeDetectionModel:
    """Deep learning model for deepfake detection"""
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def build_model(self):
        """Build ResNet18 model with transfer learning"""
        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Freeze early layers (keeping pre-trained features)
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # 2 classes: real/fake
        )
        
        self.model.to(self.device)
        logger.info(f"Model built on device: {self.device}")
    
    def create_dataloaders(self, train_real_dir, train_fake_dir, batch_size=32, val_split=0.2):
        """Create training and validation dataloaders"""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Minimal transform for validation
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create datasets
        real_dataset = DeepfakeDataset(train_real_dir, 0, transform=train_transform)
        fake_dataset = DeepfakeDataset(train_fake_dir, 1, transform=train_transform)
        
        # Combine datasets
        combined_dataset = torch.utils.data.ConcatDataset([real_dataset, fake_dataset])
        
        # Split into train/val
        val_size = int(len(combined_dataset) * val_split)
        train_size = len(combined_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            combined_dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=20, lr=0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("models/deepfake_resnet18.pth")
    
    def save_model(self, path):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights"""
        if not os.path.exists(path):
            logger.error(f"Model not found at {path}")
            return False
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
        return True
    
    def predict(self, image_array):
        """
        Predict if image is real or fake
        
        Args:
            image_array: Image as numpy array (H, W, 3)
        
        Returns:
            prediction: 0 = REAL, 1 = FAKE
            confidence: Confidence score (0-1)
        """
        self.model.eval()
        
        # Preprocess
        if image_array.max() > 1:
            image_array = image_array / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        
        # Resize
        image_tensor = transforms.Resize((256, 256))(image_tensor)
        
        # Normalize
        image_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image_tensor)
        
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return prediction, probabilities[0].cpu().numpy()


def main():
    """Main training pipeline"""
    import sys
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use augmented data if available, otherwise use original
    train_real = os.path.join(base_dir, 'data', 'train_real_augmented')
    train_fake = os.path.join(base_dir, 'data', 'train_fake_augmented')
    
    if not os.path.exists(train_real):
        train_real = os.path.join(base_dir, 'data', 'train_real')
        train_fake = os.path.join(base_dir, 'data', 'train_fake')
        print("⚠️  Augmented data not found. Using original data.")
        print("💡 Run: python augment_data.py")
    
    if not os.path.exists(train_real):
        print(f"❌ Training data not found at {train_real}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🧠 DEEP LEARNING MODEL TRAINING")
    print("="*60 + "\n")
    
    # Initialize model
    model = DeepfakeDetectionModel()
    model.build_model()
    
    # Create dataloaders
    train_loader, val_loader = model.create_dataloaders(train_real, train_fake, batch_size=16)
    
    # Train
    model.train(train_loader, val_loader, epochs=25, lr=0.001)
    
    print("\n✅ Training complete!")
    print(f"📊 Best validation accuracy: {max(model.history['val_acc']):.4f}")


if __name__ == "__main__":
    main()
