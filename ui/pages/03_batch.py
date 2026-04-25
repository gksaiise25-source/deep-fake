"""VeriFace AI — Batch Detection Page (no database)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import io
import zipfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

css_path = Path(__file__).parent.parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{open(css_path).read()}</style>", unsafe_allow_html=True)

from core.ensemble import get_detector
from core.face_analyzer import get_face_analyzer
from core.metadata_checker import analyze_metadata
from core.explainer import generate_explanation
from utils.preprocessing import load_image_rgb, save_temp_file
from utils.pdf_report import generate_pdf_report

st.markdown("## 📦 Batch Detection")
st.markdown("<p style='color:#64748b'>Analyze up to 20 images at once. Export results as CSV + ZIP of PDF reports.</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_files = st.file_uploader(
    "Upload up to 20 images",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    key="batch_upload"
)

if uploaded_files:
    files = uploaded_files[:20]
    st.markdown(f"<p style='color:#00f5ff;font-weight:600;'>{len(files)} file(s) selected</p>", unsafe_allow_html=True)

    if st.button("🚀 Run Batch Analysis", use_container_width=True):
        detector = get_detector()
        face_analyzer = get_face_analyzer()
        results = []
        pdf_bundle = {}

        progress = st.progress(0, "Starting...")
        status = st.empty()

        for i, f in enumerate(files):
            status.markdown(f"<p style='color:#94a3b8;font-size:0.85rem;'>Analyzing `{f.name}` ({i+1}/{len(files)})</p>", unsafe_allow_html=True)
            img_bytes = f.read()
            image = load_image_rgb(img_bytes)

            if image is None:
                results.append({'Filename': f.name, 'Status': 'Error', 'Verdict': '-', 'Confidence': '-', 'Fake Score': '-', 'Risk': '-'})
                progress.progress((i + 1) / len(files))
                continue

            tmp = save_temp_file(img_bytes, Path(f.name).suffix or ".jpg")
            try:
                ensemble_result = detector.analyze(image)
                faces = face_analyzer.detect_faces(image)
                face_result = face_analyzer.analyze_inconsistencies(image, faces)
                metadata_result = analyze_metadata(tmp)
                explanation = generate_explanation(ensemble_result, face_result, metadata_result)

                results.append({
                    'Filename': f.name,
                    'Status': 'Done',
                    'Verdict': ensemble_result['label'],
                    'Confidence': f"{ensemble_result['confidence']:.1f}%",
                    'Fake Score': f"{ensemble_result['deepfake_percentage']:.1f}%",
                    'Risk': explanation.get('risk_level', 'N/A'),
                    'Face Detected': 'Yes' if face_result.get('face_detected') else 'No',
                    'Time (ms)': f"{ensemble_result.get('inference_time_ms', 0):.0f}",
                })

                pdf = generate_pdf_report(ensemble_result, face_result, metadata_result, explanation)
                if pdf:
                    pdf_bundle[f.name.rsplit('.', 1)[0] + '_report.pdf'] = pdf

            except Exception as e:
                results.append({'Filename': f.name, 'Status': f'Error: {str(e)[:40]}', 'Verdict': '-', 'Confidence': '-', 'Fake Score': '-', 'Risk': '-'})
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

            progress.progress((i + 1) / len(files))

        status.empty()
        progress.empty()
        st.success(f"✅ Batch analysis complete — {len(files)} files processed")
        st.markdown("<br>", unsafe_allow_html=True)

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fake_n = sum(1 for r in results if r.get('Verdict') == 'DEEPFAKE')
        real_n = sum(1 for r in results if r.get('Verdict') == 'AUTHENTIC')
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("⚠️ Deepfakes Found", fake_n)
        sc2.metric("✅ Authentic", real_n)
        sc3.metric("❌ Errors", len(results) - fake_n - real_n)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("📥 Download Results CSV", data=df.to_csv(index=False).encode(),
            file_name="batch_results.csv", mime="text/csv", use_container_width=True)

        if pdf_bundle:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname, pdf_bytes in pdf_bundle.items():
                    zf.writestr(fname, pdf_bytes)
            st.download_button("📦 Download All PDF Reports (ZIP)", data=zip_buf.getvalue(),
                file_name="verifaceai_reports.zip", mime="application/zip", use_container_width=True)
