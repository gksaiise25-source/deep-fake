"""
Configuration file for Deepfake Detection System

Modify these settings to customize the system behavior
"""

# ============================================
# MODEL CONFIGURATION
# ============================================

# Available models: 'efficientnet', 'resnet', 'xception', 'custom'
DEFAULT_MODEL = 'efficientnet'

# Input image size (height, width)
INPUT_IMAGE_SIZE = (256, 256)

# Number of image channels (3 for RGB)
INPUT_CHANNELS = 3

# ============================================
# TRAINING CONFIGURATION
# ============================================

# Default number of training epochs
DEFAULT_EPOCHS = 50

# Default batch size
DEFAULT_BATCH_SIZE = 32

# Test set size fraction (0.2 = 20%)
TEST_SIZE = 0.2

# Validation set size fraction (0.1 = 10% of training data)
VAL_SIZE = 0.1

# Initial learning rate
INITIAL_LEARNING_RATE = 0.001

# Fine-tuning learning rate
FINE_TUNING_LEARNING_RATE = 0.0001

# Random seed for reproducibility
RANDOM_SEED = 42

# Enable data augmentation
USE_DATA_AUGMENTATION = True

# ============================================
# DATA AUGMENTATION SETTINGS
# ============================================

# Rotation range in degrees
ROTATION_RANGE = 20

# Width shift range (fraction of image width)
WIDTH_SHIFT_RANGE = 0.2

# Height shift range (fraction of image height)
HEIGHT_SHIFT_RANGE = 0.2

# Horizontal flip: True/False
HORIZONTAL_FLIP = True

# Zoom range
ZOOM_RANGE = 0.2

# Shear range
SHEAR_RANGE = 0.15

# ============================================
# FACE DETECTION SETTINGS
# ============================================

# Face detection method: 'opencv' or 'dlib'
FACE_DETECTION_METHOD = 'opencv'

# Padding around detected face (0.1 = 10%)
FACE_PADDING = 0.1

# Minimum face size (pixels)
MIN_FACE_SIZE = 30

# ============================================
# VIDEO PROCESSING SETTINGS
# ============================================

# Sample every nth frame (5 = every 5th frame)
VIDEO_SAMPLE_RATE = 5

# Maximum frames to process per video
MAX_FRAMES_PER_VIDEO = 50

# ============================================
# DATASET SETTINGS
# ============================================

# Real images directory
REAL_IMAGES_DIR = 'data/train_real'

# Fake images directory
FAKE_IMAGES_DIR = 'data/train_fake'

# Maximum images per class (None = no limit)
MAX_IMAGES_PER_CLASS = None

# ============================================
# MODEL SAVING SETTINGS
# ============================================

# Model directory
MODELS_DIR = 'models'

# Output directory for results
OUTPUTS_DIR = 'outputs'

# TensorBoard logs directory
LOGS_DIR = 'logs'

# ============================================
# CALLBACK SETTINGS
# ============================================

# Early stopping patience (epochs without improvement)
EARLY_STOPPING_PATIENCE = 5

# Checkpoint only best model
SAVE_BEST_ONLY = True

# Learning rate reduction factor
LR_REDUCTION_FACTOR = 0.5

# Learning rate reduction patience
LR_REDUCTION_PATIENCE = 3

# Minimum learning rate
MIN_LEARNING_RATE = 1e-7

# ============================================
# PREDICTION SETTINGS
# ============================================

# Prediction confidence threshold (0-1)
PREDICTION_THRESHOLD = 0.5

# Minimum confidence to display result
MIN_CONFIDENCE_DISPLAY = 0.0

# ============================================
# STREAMLIT APP SETTINGS
# ============================================

# Default model for web app
DEFAULT_WEB_MODEL = 'models/deepfake_efficientnet_final.h5'

# Maximum file upload size in MB
MAX_UPLOAD_SIZE_MB = 500

# Enable face detection by default
ENABLE_FACE_DETECTION_DEFAULT = True

# Show visualizations by default
SHOW_VISUALIZATION_DEFAULT = True

# ============================================
# LOGGING SETTINGS
# ============================================

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Log format
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ============================================
# PERFORMANCE SETTINGS
# ============================================

# Enable GPU acceleration
USE_GPU = True

# Mixed precision training (for faster training)
MIXED_PRECISION = False

# Number of worker threads for data loading
NUM_WORKERS = 4

# ============================================
# EVALUATION METRICS
# ============================================

# Metrics to track during training
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'auc'
]

# Display metrics in verbose mode
VERBOSE_MODE = 1

# ============================================
# DEPLOYMENT SETTINGS
# ============================================

# Model format: 'h5' or 'savedmodel'
MODEL_FORMAT = 'h5'

# Quantization for model compression (future feature)
QUANTIZATION = False

# ============================================
# ADVANCED SETTINGS
# ============================================

# Unfreeze base model layers for fine-tuning
UNFREEZE_BASE_MODEL = False

# Number of base model layers to unfreeze
UNFREEZE_LAYERS_COUNT = 20

# Use class weights for imbalanced data
USE_CLASS_WEIGHTS = False

# ============================================

def get_config(key: str, default=None):
    """
    Get configuration value
    
    Args:
        key (str): Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    import sys
    module = sys.modules[__name__]
    return getattr(module, key, default)


def print_config():
    """Print all configuration settings"""
    import sys
    
    module = sys.modules[__name__]
    
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION SYSTEM - CONFIGURATION")
    print("="*60 + "\n")
    
    # Get all module attributes that are configuration settings
    settings = {k: v for k, v in vars(module).items() 
               if not k.startswith('_') and k.isupper()}
    
    # Group settings by section (comments with ===)
    for key, value in sorted(settings.items()):
        print(f"{key}: {value}")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    print_config()
