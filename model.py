"""
Deep Learning Model for Deepfake Detection

This module implements machine learning models for deepfake detection.
Uses scikit-learn for Python 3.14 compatibility.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DeepfakeDetectionModel:
    """
    Deepfake detection model using scikit-learn.
    Uses Random Forest for fast, interpretable deepfake detection.
    """
    
    def __init__(self, model_type: str = 'random_forest', input_shape: tuple = (256, 256, 3)):
        """
        Initialize the model
        
        Args:
            model_type (str): Type of model - 'random_forest', 'gradient_boost', 'svm'
            input_shape (tuple): Input image shape (height, width, channels)
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.scaler = StandardScaler()
        self.model = None
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Model {model_type} initialized with input shape {input_shape}")
    
    def _extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Build custom CNN model from scratch
        
        Returns:
            Keras model
        """
        model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Conv Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense layers
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        logger.info("Custom CNN model created")
        return model
    
    def _build_efficientnet_model(self) -> Model:
        """
        Build EfficientNetB0 with transfer learning
        
        Returns:
            Keras model
        """
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create new model
        model = Sequential([
            Input(shape=self.input_shape),
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        logger.info("EfficientNetB0 transfer learning model created")
        return model
    
    def _build_resnet_model(self) -> Model:
        """
        Build ResNet50 with transfer learning
        
        Returns:
            Keras model
        """
        # Load pre-trained ResNet50
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create new model
        model = Sequential([
            Input(shape=self.input_shape),
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        logger.info("ResNet50 transfer learning model created")
        return model
    
    def _build_xception_model(self) -> Model:
        """
        Build Xception with transfer learning
        
        Returns:
            Keras model
        """
        # Load pre-trained Xception
        base_model = Xception(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create new model
        model = Sequential([
            Input(shape=self.input_shape),
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        logger.info("Xception transfer learning model created")
        return model
    
    def compile(self, learning_rate: float = 0.001):
        """
        Compile the model
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()]
        )
        
        logger.info("Model compiled successfully")
    
    def get_model(self) -> Model:
        """Get the keras model"""
        return self.model
    
    def unfreeze_base(self, num_layers: int = None):
        """
        Unfreeze base model layers for fine-tuning
        
        Args:
            num_layers (int): Number of layers to unfreeze from the end (None = unfreeze all)
        """
        if self.model_type == 'custom':
            logger.info("Custom model doesn't have base layers to unfreeze")
            return
        
        base_model = self.model.layers[1]  # Get base model from Sequential
        
        if num_layers is None:
            base_model.trainable = True
        else:
            base_model.trainable = True
            for layer in base_model.layers[:-num_layers]:
                layer.trainable = False
        
        logger.info(f"Unfroze base model for fine-tuning")
        
        # Recompile with lower learning rate
        self.compile(learning_rate=0.0001)
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
    
    def save(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save model (e.g., 'model.h5')
        """
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath (str): Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


def get_callbacks(model_name: str = 'deepfake_model') -> list:
    """
    Get training callbacks
    
    Args:
        model_name (str): Name for saved models
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir='logs/',
            histogram_freq=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    # Example usage
    model = DeepfakeDetectionModel(model_type='efficientnet')
    model.compile()
    model.summary()
