# Deepfake Detection System - Quick Start Guide

## ⚡ 5-Minute Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Sample Dataset
```bash
python train.py --create-sample
```

### 3. Train Model
```bash
python train.py --create-sample --epochs 20 --batch-size 16
```

### 4. Launch Web App
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser!

---

## 📋 Full Command Reference

### Training
```bash
# Quick training with sample data
python train.py --create-sample

# Full training with your dataset
python train.py --real-dir data/train_real --fake-dir data/train_fake --epochs 100

# Different models
python train.py --model efficientnet
python train.py --model resnet
python train.py --model xception
python train.py --model custom

# Advanced options
python train.py --epochs 100 --batch-size 64 --max-images 5000 --no-augmentation
```

### Prediction
```bash
# Predict on image
python predict.py --image path/to/image.jpg --output result.png

# Predict on video
python predict.py --video path/to/video.mp4

# Custom model
python predict.py --image photo.jpg --model models/deepfake_resnet_final.h5

# Disable face detection
python predict.py --image photo.jpg --no-face-detection
```

### Testing & Configuration
```bash
# Run tests
python test.py

# Setup environment
python setup.py

# View configuration
python config.py
```

### Web Interface
```bash
# Launch Streamlit app
streamlit run app.py

# Custom port
streamlit run app.py -- --server.port 8502

# Public sharing
streamlit run app.py -- --logger.level=debug
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Test
```bash
python train.py --create-sample --epochs 5
python predict.py --image data/train_real/sample_real_000.jpg
```

### Workflow 2: Full Training
```bash
# Organize your data in data/train_real and data/train_fake
python train.py --epochs 100 --batch-size 32
```

### Workflow 3: Web Interface
```bash
python train.py --create-sample --epochs 20
streamlit run app.py
# Upload images/videos in the web interface
```

### Workflow 4: Batch Processing
```bash
# Create a Python script with:
from predict import DeepfakePredictor
predictor = DeepfakePredictor('models/deepfake_efficientnet_final.h5')
results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

---

## 📊 Expected Performance

- **Quick Test (5 epochs)**: 
  - Accuracy: ~60-70%
  - Training time: 5-10 minutes

- **Standard Training (50 epochs)**:
  - Accuracy: ~85-90%
  - Training time: 1-2 hours

- **Extended Training (100+ epochs)**:
  - Accuracy: ~92-96%
  - Training time: 3-5 hours

*Times vary based on hardware and dataset size*

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'tensorflow'` | Run `pip install -r requirements.txt` |
| Out of memory | Use `--batch-size 16` and `--max-images 500` |
| Model not found | Train first: `python train.py --create-sample` |
| No faces detected | Use `--no-face-detection` flag |
| Slow training | Enable GPU or reduce image size |

---

## 📁 After Training, You'll Have:

```
models/
├── deepfake_efficientnet_best.h5      # Best model
└── deepfake_efficientnet_final.h5     # Final model

outputs/
├── results.json                        # Evaluation metrics
├── training_history.json               # Training progression
└── visualizations/                     # Generated plots

logs/
└── (TensorBoard logs)
```

---

## 🌐 Web App Features

1. **Upload Image/Video**: Drag & drop support
2. **Real-time Analysis**: Instant predictions
3. **Visualization**: See analysis charts
4. **Model Selection**: Choose from 4 models
5. **Confidence Scores**: Know how sure the model is
6. **Detailed Results**: Per-frame analysis for videos

---

## 💡 Tips for Best Results

1. **Data Quality**: Use clear, high-resolution images
2. **Data Balance**: Equal real and fake images
3. **Diversity**: Include various angles and lighting
4. **Preprocessing**: Text on images okay, very small faces problematic
5. **Augmentation**: Enable for better generalization
6. **Batch Size**: Start with 32, increase if GPU available
7. **Epochs**: More epochs = better, but watch for overfitting

---

See README.md for complete documentation!
