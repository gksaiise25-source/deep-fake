"""VeriFace AI — Scan History Page (no database — session-based)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
from pathlib import Path

css_path = Path(__file__).parent.parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{open(css_path).read()}</style>", unsafe_allow_html=True)

st.markdown("## 📜 Scan History")
st.markdown("<p style='color:#64748b'>Scans from this session are listed below.</p>", unsafe_allow_html=True)
st.markdown("---")

# Session-state scan log (populated by the analyze page)
scans = st.session_state.get('scan_log', [])

if not scans:
    st.markdown("""
    <div style="text-align:center;padding:60px;background:rgba(255,255,255,0.02);border:1px solid rgba(0,245,255,0.1);border-radius:16px;">
        <div style="font-size:3rem;margin-bottom:12px;">🗂️</div>
        <div style="color:#64748b;font-size:1rem;">No scans yet this session. Run an analysis in <strong>🔍 Analyze Media</strong>!</div>
    </div>""", unsafe_allow_html=True)
else:
    total = len(scans)
    fake_n = sum(1 for s in scans if s.get('is_fake'))
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Total Scans", total)
    c2.metric("⚠️ Deepfakes", fake_n)
    c3.metric("✅ Authentic", total - fake_n)
    st.markdown("<br>", unsafe_allow_html=True)

    df = pd.DataFrame(scans)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("📥 Export CSV", data=df.to_csv(index=False).encode(),
        file_name="verifaceai_history.csv", mime="text/csv", use_container_width=True)
