# Deployment Guide - Deepfake Detection System

## Quick Deployment to Streamlit Cloud (Recommended)

Streamlit Cloud is the easiest way to deploy. It's **free**, **reliable**, and takes about 5 minutes.

### Prerequisites
- GitHub account (https://github.com)
- Streamlit Cloud account (https://streamlit.io/cloud)

### Step 1: Prepare Your GitHub Repository

If you haven't already created a Git repo locally:

```bash
cd deepfake_detection
git init
git add .
git commit -m "Initial commit: Deepfake detection system"
```

### Step 2: Push Code to GitHub

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Name it: `deepfake-detection` (or your preferred name)
   - Click "Create repository"

2. **Add remote and push your code:**

```bash
git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:** https://share.streamlit.io/

2. **Sign in with GitHub** when prompted

3. **Click "New app" button**

4. **Fill in the deployment form:**
   - **Repository:** `YOUR_USERNAME/deepfake-detection`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`

5. **Click "Deploy"** ✅

The app will be deployed at: `https://YOUR_USERNAME-deepfake-detection.streamlit.app/`

---

## Deployment on Other Platforms

### Heroku Deployment

1. **Create a new file: `Procfile`**

```
web: streamlit run streamlit_app.py
```

2. **Create a new file: `.env`** (for environment variables)

```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8080
```

3. **Deploy:**

```bash
heroku login
heroku create deepfake-detection
git push heroku main
```

### AWS/GCP/Azure

For enterprise deployments, use Docker:

1. **Create `Dockerfile`:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

2. **Build and deploy:**

```bash
docker build -t deepfake-detection .
docker run -p 8080:8080 deepfake-detection
```

---

## Important Notes

### Model Size
- Current model file: `models/deepfake_advanced_ra.pkl` (~50-200MB)
- Streamlit Cloud has file size limits, ensure models are < 500MB
- If larger, consider using cloud storage (AWS S3, Google Cloud Storage)

### Dependencies
- All packages in `requirements.txt` are compatible with Streamlit Cloud
- No GPU required (CPU inference works fine)
- Expects Python 3.9+

### Performance Optimization
- First load may be slow due to model initialization
- Use `@st.cache_resource` to cache model (already implemented)
- Consider lazy loading for large models

### Security
- Keep `secrets.toml` file for any API keys (not committed)
- Use environment variables for sensitive data
- GitHub repo should be public for free deployment

---

## Troubleshooting

### "ModuleNotFoundError: No module named..."
- Check all imports in `requirements.txt`
- Ensure all Python files are in the repo

### Model not found error
- Verify model files are in `models/` folder
- Check `streamlit_app.py` paths are relative, not absolute

### App crashes on startup
- Check Streamlit logs in the cloud dashboard
- Test locally first: `streamlit run streamlit_app.py`

### Slow startup
- Model loading is cached on first run
- Subsequent refreshes will be faster

---

## Monitoring After Deployment

1. **View logs:**
   - Go to your app's Streamlit Cloud dashboard
   - Click "Manage app" → "View logs"

2. **Monitor performance:**
   - Track usage statistics
   - Check for errors in real-time

3. **Update code:**
   - Make changes locally
   - Push to GitHub: `git push`
   - Changes auto-deploy within seconds

---

## Next Steps

After deployment:

1. **Test the live app** in production
2. **Train on real datasets** for better accuracy
3. **Add more features** (batch processing, API endpoints)
4. **Monitor usage** and user feedback
5. **Continuously improve** the model

---

## Support

For issues:
- Streamlit Cloud: https://docs.streamlit.io/streamlit-cloud
- Streamlit Forum: https://discuss.streamlit.io/
- GitHub Issues: Create an issue in your repository
