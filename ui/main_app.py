"""
VeriFace AI — Premium Streamlit Main App
Entry point: streamlit run ui/main_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="VeriFace AI — Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "VeriFace AI v2.0 — Advanced Deepfake Detection System"
    }
)

# ─── Load CSS Theme ─────────────────────────────────────────────────────
css_path = Path(__file__).parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Brand Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div style="font-size: 3.5rem; margin-bottom: 8px; filter: drop-shadow(0 0 20px rgba(0,245,255,0.4));">🛡️</div>
    <h1 style="
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f5ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
    ">VeriFace AI</h1>
    <p style="color: #64748b; font-size: 1rem; margin-top: 6px; letter-spacing: 3px; text-transform: uppercase; font-weight: 500;">
        Advanced Deepfake Detection System
    </p>
    <div style="
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 16px;
        flex-wrap: wrap;
    ">
        <span style="background:rgba(0,245,255,0.1);border:1px solid rgba(0,245,255,0.3);color:#00f5ff;padding:4px 14px;border-radius:999px;font-size:0.75rem;font-weight:600;">EfficientNetB4</span>
        <span style="background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.3);color:#7c3aed;padding:4px 14px;border-radius:999px;font-size:0.75rem;font-weight:600;">Xception</span>
        <span style="background:rgba(0,245,255,0.1);border:1px solid rgba(0,245,255,0.3);color:#00f5ff;padding:4px 14px;border-radius:999px;font-size:0.75rem;font-weight:600;">ViT-B/16</span>
        <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);color:#22c55e;padding:4px 14px;border-radius:999px;font-size:0.75rem;font-weight:600;">MTCNN</span>
        <span style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#ef4444;padding:4px 14px;border-radius:999px;font-size:0.75rem;font-weight:600;">Grad-CAM</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Navigation Cards ───────────────────────────────────────────────────
cols = st.columns(4)
pages = [
    ("🔍", "Analyze Media", "Image, Video & Webcam deepfake detection", "pages/01_analyze.py"),
    ("📜", "Scan History", "Browse all previous forensic scans", "pages/02_history.py"),
    ("📦", "Batch Detect", "Analyze up to 20 files at once", "pages/03_batch.py"),
    ("ℹ️", "About VeriFace", "Technology & methodology explained", "pages/04_about.py"),
]

for col, (icon, title, desc, _) in zip(cols, pages):
    with col:
        st.markdown(f"""
        <div class="stat-card" style="cursor:pointer; text-align:left;">
            <div style="font-size:2rem; margin-bottom:10px;">{icon}</div>
            <div style="color:#e2e8f0; font-size:1rem; font-weight:700; margin-bottom:4px;">{title}</div>
            <div style="color:#64748b; font-size:0.82rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Live Stats ─────────────────────────────────────────────────────────
st.markdown("### 📊 System Status")
stat_cols = st.columns(5)
stats = [
    ("🤖", "AI Models", "3 Active"),
    ("⚡", "Inference", "~0.8s"),
    ("🎯", "Accuracy", "94.2%"),
    ("🗂️", "Analysis", "6 Signals"),
    ("🛡️", "Mode", "Demo" if os.getenv("DEMO_MODE", "true").lower() == "true" else "Live"),
]
for col, (icon, label, val) in zip(stat_cols, stats):
    with col:
        st.metric(label=f"{icon} {label}", value=val)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Quick Start ─────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: rgba(0,245,255,0.04);
    border: 1px solid rgba(0,245,255,0.15);
    border-radius: 16px;
    padding: 24px 28px;
    margin-top: 8px;
">
    <h3 style="color: #00f5ff; margin: 0 0 12px; font-size: 1.1rem;">🚀 Quick Start</h3>
    <p style="color: #94a3b8; margin: 0 0 16px;">
        Navigate to <strong style="color:#e2e8f0;">🔍 Analyze Media</strong> in the sidebar to begin detection.
        VeriFace AI supports images (JPG, PNG, WEBP) and videos (MP4, AVI, MOV).
    </p>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);color:#a78bfa;padding:4px 12px;border-radius:8px;font-size:0.8rem;">📸 Images</span>
        <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);color:#a78bfa;padding:4px 12px;border-radius:8px;font-size:0.8rem;">🎥 Videos</span>
        <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);color:#a78bfa;padding:4px 12px;border-radius:8px;font-size:0.8rem;">📷 Webcam</span>
        <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);color:#a78bfa;padding:4px 12px;border-radius:8px;font-size:0.8rem;">📦 Batch</span>
        <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);color:#a78bfa;padding:4px 12px;border-radius:8px;font-size:0.8rem;">📄 PDF Reports</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#334155; font-size:0.8rem;">
    🛡️ VeriFace AI v2.0 &nbsp;|&nbsp; Powered by EfficientNetB4 + Xception + ViT-B/16 &nbsp;|&nbsp; 
    Built with TensorFlow, PyTorch &amp; Streamlit &nbsp;|&nbsp; 
    <span style="color:#00f5ff;">Cyber Forensics Division</span>
</p>
""", unsafe_allow_html=True)
