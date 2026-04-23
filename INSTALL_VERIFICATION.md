# Deepfake Detection System - Setup Verification

This file contains verification steps and important information

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Tests pass: `python test.py`
- [ ] Models directory created: `models/`
- [ ] Data directory created: `data/`
- [ ] Outputs directory created: `outputs/`

## 🔍 Verification Commands

Run these commands to verify installation:

```bash
# Check Python version
python --version

# Test imports
python -c "import tensorflow; print('TensorFlow OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import numpy; print('NumPy OK')"

# Run test suite
python test.py

# Check directories
dir models\
dir data\
dir outputs\
```

## 🚀 Quick Test Route

```bash
# 1. Setup (if not done)
python setup.py

# 2. Create sample data
python train.py --create-sample

# 3. Train model quickly
python train.py --create-sample --epochs 5 --batch-size 8

# 4. Test prediction
python predict.py --image data/train_real/sample_real_000.jpg

# 5. Launch web app
streamlit run app.py
```

## 📊 Expected Results

### After quick training:
- Training loss: ~0.5-0.3
- Validation accuracy: ~60-70%
- Duration: 5-10 minutes

### After full training:
- Training loss: ~0.1-0.05
- Validation accuracy: ~90-96%
- Duration: 2-4 hours (depends on data)

## 📁 Final Directory Structure

```
deepfake_detection/
├── data/
│   ├── train_real/        (optional - for your data)
│   ├── train_fake/        (optional - for your data)
│   └── (sample files if --create-sample used)
├── models/
│   └── (trained .h5 files after training)
├── outputs/
│   ├── results.json       (metrics after training)
│   └── training_history.json
├── logs/                  (TensorBoard logs)
└── [all Python files]
```

## 🔗 Important Links

- **TensorFlow**: https://www.tensorflow.org/
- **OpenCV**: https://opencv.org/
- **Streamlit**: https://streamlit.io/
- **Keras**: https://keras.io/

## 📞 Troubleshooting Quick Reference

| Error | Fix |
|-------|-----|
| `No module named 'tensorflow'` | `pip install tensorflow` |
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'streamlit'` | `pip install streamlit` |
| Out of memory | Reduce batch size or max images |
| CUDA not found | Use CPU (will be slower) |

## ✨ You're All Set!

Everything is ready. Choose your next action:

1. **Want quick demo?** → `python train.py --create-sample --epochs 5`
2. **Want web app?** → `streamlit run app.py`
3. **Want to use your data?** → Prepare `data/train_real` and `data/train_fake`
4. **Want to learn more?** → Read `README.md` and `QUICKSTART.md`

---

**Happy detecting! 🔍**
