"""
VeriFace AI — Analyze Media Page
Image / Video / Webcam detection — no database required.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from core.ensemble import get_detector
from core.face_analyzer import get_face_analyzer
from core.gradcam import generate_region_heatmap
from core.metadata_checker import analyze_metadata
from core.explainer import generate_explanation
from utils.preprocessing import load_image_rgb, save_temp_file
from utils.pdf_report import generate_pdf_report

css_path = Path(__file__).parent.parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{open(css_path).read()}</style>", unsafe_allow_html=True)

st.markdown("## 🔍 Analyze Media")
st.markdown("<p style='color:#64748b'>Upload an image or video to begin deepfake forensic analysis.</p>", unsafe_allow_html=True)
st.markdown("---")

tab_img, tab_vid, tab_cam = st.tabs(["📸 Image", "🎥 Video", "📷 Live Webcam"])


def render_result(ensemble_result, face_result, metadata_result, explanation, image=None, heatmap=None):
    is_fake = ensemble_result['is_fake']
    conf = ensemble_result['confidence']
    fake_pct = ensemble_result['deepfake_percentage']
    demo_mode = ensemble_result.get('is_demo_mode', True)

    if demo_mode:
        st.info("ℹ️ **Demo Mode** — PyTorch not installed or model weights missing. Install PyTorch for real inference.")

    if is_fake:
        st.markdown(f"""
        <div class="verdict-fake">
            <div style="font-size:2rem;font-weight:800;color:#ef4444;">⚠️ DEEPFAKE DETECTED</div>
            <div style="color:#fca5a5;margin-top:4px;">This content shows strong indicators of AI manipulation</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-real">
            <div style="font-size:2rem;font-weight:800;color:#22c55e;">✅ CONTENT AUTHENTIC</div>
            <div style="color:#86efac;margin-top:4px;">No deepfake artifacts detected in this content</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Confidence", f"{conf:.1f}%")
    c2.metric("🔥 Fake Score", f"{fake_pct:.1f}%")
    c3.metric("⏱️ Time", f"{ensemble_result.get('inference_time_ms', 0):.0f}ms")
    c4.metric("👤 Face Found", "Yes" if face_result.get('face_detected') else "No")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1])

    with col_l:
        fill_class = "confidence-fill-fake" if is_fake else "confidence-fill-real"
        st.markdown(f"""
        <div style="margin-bottom:20px;">
            <div style="color:#94a3b8;font-size:0.82rem;margin-bottom:6px;font-weight:600;letter-spacing:1px;">DEEPFAKE PROBABILITY</div>
            <div class="confidence-meter">
                <div class="{fill_class}" style="width:{fake_pct}%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px;">
                <span style="color:#22c55e;font-size:0.73rem;font-weight:600;">AUTHENTIC</span>
                <span style="color:#ef4444;font-size:0.73rem;font-weight:600;">DEEPFAKE</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='color:#94a3b8;font-size:0.82rem;margin-bottom:10px;font-weight:600;letter-spacing:1px;'>MODEL SCORES</div>", unsafe_allow_html=True)
        for model_name, score in ensemble_result.get('per_model_scores', {}).items():
            st.markdown(f"""
            <div class="model-score-bar">
                <div class="model-score-label">{model_name}</div>
                <div class="model-score-track">
                    <div class="model-score-fill" style="width:{score}%"></div>
                </div>
                <div class="model-score-value">{score:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        if heatmap is not None:
            st.image(heatmap, caption="🔥 Activation Heatmap — Red = suspicious regions", use_container_width=True)
        elif image is not None:
            st.image(image, caption="Analyzed Image", use_container_width=True)

    # AI Explanation
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🤖 AI Forensic Explanation", expanded=True):
        st.markdown(f"<p style='color:#e2e8f0;line-height:1.7;'>{explanation.get('verdict_summary', '')}</p>", unsafe_allow_html=True)
        st.markdown("**Key Findings:**")
        for finding in explanation.get('key_findings', []):
            st.markdown(finding)
        rec = explanation.get('recommendation', '')
        if rec:
            st.markdown(f"<br><p style='color:#94a3b8;'>{rec}</p>", unsafe_allow_html=True)

    # Face Inconsistency
    incon = face_result.get('inconsistency_scores', {})
    if incon and face_result.get('face_detected'):
        with st.expander("👤 Facial Inconsistency Signals"):
            fc = st.columns(3)
            labels = {
                'skin_texture': ('🧴', 'Skin Texture'),
                'boundary_artifacts': ('⚡', 'Boundary Artifacts'),
                'lighting_inconsistency': ('💡', 'Lighting'),
                'eye_anomaly': ('👁️', 'Eye Anomaly'),
                'color_inconsistency': ('🎨', 'Color Stats'),
                'frequency_artifacts': ('📡', 'Freq Fingerprint'),
            }
            for i, (k, (icon, lbl)) in enumerate(labels.items()):
                with fc[i % 3]:
                    val = incon.get(k, 0)
                    color = "#ef4444" if val > 60 else "#f59e0b" if val > 35 else "#22c55e"
                    st.markdown(f"""
                    <div class="stat-card" style="text-align:center;margin-bottom:12px;padding:14px;">
                        <div style="font-size:0.75rem;color:#64748b;margin-bottom:4px;">{icon} {lbl}</div>
                        <div style="font-size:1.5rem;font-weight:700;color:{color};">{val:.0f}</div>
                        <div style="font-size:0.65rem;color:#475569;">/100 risk</div>
                    </div>""", unsafe_allow_html=True)

    # Metadata
    with st.expander("🗂️ Metadata Forensics"):
        m1, m2 = st.columns(2)
        with m1:
            st.write(f"**File:** `{metadata_result.get('file_name', 'N/A')}`")
            st.write(f"**Size:** `{metadata_result.get('file_size_kb', 0):.1f} KB`")
            st.write(f"**EXIF:** `{'Present' if metadata_result.get('exif_present') else 'Missing'}`")
            st.write(f"**Camera:** `{metadata_result.get('camera_info') or 'Not found'}`")
        with m2:
            st.write(f"**Tampering Score:** `{metadata_result.get('tampering_score', 0)}/100`")
            st.write(f"**Verdict:** `{metadata_result.get('tampering_verdict', 'N/A')}`")
            st.write(f"**GAN Signature:** `{'⚠️ FOUND' if metadata_result.get('gan_signature_found') else 'Not found'}`")
        for sig in metadata_result.get('tampering_signals', []):
            st.markdown(f"• {sig}")

    # PDF Download
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_bytes = generate_pdf_report(ensemble_result, face_result, metadata_result, explanation)
    if pdf_bytes:
        st.download_button(
            label="📄 Download PDF Forensic Report",
            data=pdf_bytes,
            file_name=f"verifaceai_report_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ─── IMAGE TAB ──────────────────────────────────────────────────────────────
with tab_img:
    uploaded = st.file_uploader(
        "Drag & drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
        key="img_upload"
    )

    if uploaded:
        image_bytes = uploaded.read()
        image = load_image_rgb(image_bytes)

        if image is not None:
            col_prev, col_info = st.columns([2, 1])
            with col_prev:
                st.image(image, caption=f"📁 {uploaded.name}", use_container_width=True)
            with col_info:
                st.markdown(f"**File:** `{uploaded.name}`")
                st.markdown(f"**Size:** `{len(image_bytes)/1024:.1f} KB`")
                st.markdown(f"**Dimensions:** `{image.shape[1]}×{image.shape[0]}`")
                analyze_btn = st.button("🔍 Analyze", key="analyze_img", use_container_width=True)

            if analyze_btn:
                tmp = save_temp_file(image_bytes, Path(uploaded.name).suffix or ".jpg")
                try:
                    progress = st.progress(0, "Checking metadata...")
                    # ── Step 1: metadata FIRST (feeds into ensemble) ──
                    metadata_result = analyze_metadata(tmp)
                    progress.progress(20, "Running forensic analysis...")
                    detector = get_detector()
                    # ── Step 2: ensemble with metadata context ─────────
                    ensemble_result = detector.analyze(image, metadata_result=metadata_result)
                    progress.progress(55, "Detecting face...")
                    face_analyzer = get_face_analyzer()
                    faces = face_analyzer.detect_faces(image)
                    face_result = face_analyzer.analyze_inconsistencies(image, faces)
                    # Merge face_detected from ensemble (uses Haar cascade)
                    if not face_result.get('face_detected') and ensemble_result.get('face_detected'):
                        face_result['face_detected'] = True
                    progress.progress(72, "Generating heatmap...")
                    heatmap = generate_region_heatmap(image, face_result.get('primary_face_bbox'), ensemble_result['fake_probability'])
                    progress.progress(90, "Building explanation...")
                    explanation = generate_explanation(ensemble_result, face_result, metadata_result)
                    progress.progress(100, "Done!")
                    time.sleep(0.3)
                    progress.empty()
                    st.session_state['img_results'] = (ensemble_result, face_result, metadata_result, explanation, image, heatmap)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

    if 'img_results' in st.session_state:
        st.markdown("---")
        render_result(*st.session_state['img_results'])


# ─── VIDEO TAB ──────────────────────────────────────────────────────────────
with tab_vid:
    uploaded_vid = st.file_uploader(
        "Drag & drop a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="vid_upload"
    )
    max_frames = st.slider("Max frames to analyze", 10, 80, 40, key="max_frames_slider")

    if uploaded_vid:
        st.video(uploaded_vid)
        if st.button("🔍 Analyze Video", key="analyze_vid", use_container_width=True):
            vid_bytes = uploaded_vid.read()
            tmp = save_temp_file(vid_bytes, Path(uploaded_vid.name).suffix or ".mp4")
            try:
                from core.video_analyzer import VideoAnalyzer
                progress = st.progress(0, "Initializing...")
                detector = get_detector()
                va = VideoAnalyzer(ensemble_detector=detector, max_frames=max_frames)
                result = va.analyze_video(tmp, progress_callback=lambda p: progress.progress(int(p * 90), "Analyzing frames..."))
                progress.progress(100, "Done!")
                time.sleep(0.3)
                progress.empty()

                if 'error' not in result:
                    if result['is_fake']:
                        st.markdown('<div class="verdict-fake"><div style="font-size:1.8rem;font-weight:800;color:#ef4444;">⚠️ DEEPFAKE VIDEO DETECTED</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="verdict-real"><div style="font-size:1.8rem;font-weight:800;color:#22c55e;">✅ VIDEO APPEARS AUTHENTIC</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("🎞️ Frames", result['frames_analyzed'])
                    v2.metric("🔥 Fake %", f"{result['deepfake_percentage']:.1f}%")
                    v3.metric("⏱️ Duration", f"{result['duration_seconds']:.1f}s")
                    v4.metric("🎯 Confidence", f"{result['confidence']:.1f}%")

                    try:
                        import plotly.graph_objects as go
                        probs = result.get('fake_probabilities_timeline', [])
                        if probs:
                            bar_colors = ['rgba(239,68,68,0.8)' if p > 0.5 else 'rgba(34,197,94,0.8)' for p in probs]
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=list(range(len(probs))), y=[p * 100 for p in probs], marker_color=bar_colors))
                            fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.4)", annotation_text="Threshold")
                            fig.update_layout(
                                title="Frame-by-Frame Deepfake Probability",
                                xaxis_title="Sample Frame", yaxis_title="Fake Probability (%)",
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#94a3b8', yaxis_range=[0, 100],
                                title_font_color='#e2e8f0', showlegend=False,
                                xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.markdown(f"**Frame predictions:** {result.get('fake_probabilities_timeline', [])}")

                    tc = result.get('temporal_consistency', {})
                    if tc:
                        st.markdown(f"**Temporal Consistency:** `{tc.get('interpretation', 'N/A')}` | Score: `{tc.get('score', 0):.1f}/100`")
                else:
                    st.error(f"Video analysis failed: {result['error']}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)


# ─── WEBCAM TAB ─────────────────────────────────────────────────────────────
with tab_cam:
    st.markdown("""
    <div style="background:rgba(0,245,255,0.04);border:1px solid rgba(0,245,255,0.15);border-radius:16px;padding:24px;margin-bottom:20px;">
        <h3 style="color:#00f5ff;margin:0 0 8px;">📷 Live Webcam Detection</h3>
        <p style="color:#64748b;margin:0;">Capture a photo from your webcam for instant deepfake analysis.</p>
    </div>""", unsafe_allow_html=True)

    cam_image = st.camera_input("Capture image for analysis")
    if cam_image:
        image = load_image_rgb(cam_image.getvalue())
        if image is not None:
            with st.spinner("🧠 Analyzing captured frame..."):
                detector = get_detector()
                ensemble_result = detector.analyze(image)
                face_analyzer = get_face_analyzer()
                faces = face_analyzer.detect_faces(image)
                face_result = face_analyzer.analyze_inconsistencies(image, faces)
                heatmap = generate_region_heatmap(image, face_result.get('primary_face_bbox'), ensemble_result['fake_probability'])
                metadata_result = {
                    'file_name': 'webcam_capture.jpg', 'file_size_kb': 0,
                    'exif_present': False, 'tampering_signals': [], 'tampering_score': 0,
                    'tampering_verdict': 'N/A', 'gan_signature_found': False,
                    'camera_info': 'Webcam', 'creation_date': None,
                }
                explanation = generate_explanation(ensemble_result, face_result, metadata_result)
            render_result(ensemble_result, face_result, metadata_result, explanation, image, heatmap)
