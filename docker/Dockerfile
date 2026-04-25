# VeriFace AI — Dockerfile
FROM python:3.11-slim

LABEL maintainer="VeriFace AI Team"
LABEL description="Advanced Deepfake Detection System"

WORKDIR /app

# System deps for OpenCV + audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libsndfile1 ffmpeg libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create dirs
RUN mkdir -p models data logs outputs

# Expose ports
EXPOSE 8501 8000

# Default: run Streamlit UI
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV DEMO_MODE=true

CMD ["streamlit", "run", "ui/main_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=500", \
     "--theme.base=dark"]
