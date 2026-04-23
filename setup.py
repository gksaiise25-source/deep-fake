#!/usr/bin/env python3
"""
Setup script for Deepfake Detection System

Run this script to set up the environment and create necessary directories
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    logger.info("Creating necessary directories...")
    
    directories = [
        'data/train_real',
        'data/train_fake',
        'data/test_real',
        'data/test_fake',
        'models',
        'outputs',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created {directory}/")


def check_python_version():
    """Check if Python version is compatible"""
    logger.info("Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        logger.info(f"✓ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        logger.error(f"✗ Python {version.major}.{version.minor} is not compatible (need 3.8+)")
        return False


def install_requirements():
    """Install Python requirements"""
    logger.info("Installing Python requirements...")
    
    if not os.path.exists('requirements.txt'):
        logger.error("✗ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        logger.info("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install requirements: {e}")
        return False


def verify_imports():
    """Verify critical imports"""
    logger.info("Verifying critical modules...")
    
    critical_modules = [
        'numpy',
        'cv2',
        'tensorflow',
        'sklearn',
        'matplotlib'
    ]
    
    failed = []
    
    for module in critical_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module} is available")
        except ImportError:
            logger.error(f"✗ {module} is not available")
            failed.append(module)
    
    if failed:
        logger.error(f"Failed to import: {', '.join(failed)}")
        return False
    
    return True


def create_gitignore():
    """Create .gitignore file"""
    logger.info("Creating .gitignore...")
    
    gitignore_content = """
# Data and Models
data/
models/
outputs/
logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
.env.local

# Temporary
*.tmp
tmp/
temp/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    logger.info("✓ .gitignore created")


def main():
    """Main setup function"""
    logger.info("="*60)
    logger.info("Deepfake Detection System - Setup")
    logger.info("="*60)
    
    # Check Python version
    if not check_python_version():
        logger.error("Please install Python 3.8 or higher")
        return 1
    
    # Create directories
    create_directories()
    
    # Create .gitignore
    create_gitignore()
    
    # Install requirements
    logger.info("\nInstalling dependencies (this may take a few minutes)...")
    logger.info("Make sure you have internet connection...")
    
    if not install_requirements():
        logger.error("Failed to install requirements")
        logger.info("Try manually running: pip install -r requirements.txt")
        return 1
    
    # Verify imports
    if not verify_imports():
        logger.error("Some modules failed to import")
        logger.info("Try running: pip install -r requirements.txt --upgrade")
        return 1
    
    # Success
    logger.info("\n" + "="*60)
    logger.info("✅ Setup completed successfully!")
    logger.info("="*60)
    
    logger.info("\nNext steps:")
    logger.info("1. Review config.py for configuration options")
    logger.info("2. Prepare your dataset or run: python train.py --create-sample")
    logger.info("3. Train the model: python train.py")
    logger.info("4. Launch the web app: streamlit run app.py")
    logger.info("5. Or make predictions: python predict.py --image path/to/image.jpg")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
