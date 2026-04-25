"""
VeriFace AI — AI Explanation Generator
Produces human-readable forensic explanations for detection results.
"""

from typing import Dict, List


def generate_explanation(
    ensemble_result: Dict,
    face_result: Dict,
    metadata_result: Dict,
) -> Dict:
    """
    Generate a structured forensic explanation based on all analysis results.
    Returns title, verdict_summary, key_findings list, and recommendation.
    """
    is_fake = ensemble_result.get('is_fake', False)
    confidence = ensemble_result.get('confidence', 0)
    fake_pct = ensemble_result.get('deepfake_percentage', 0)
    face_detected = face_result.get('face_detected', False)
    inconsistency = face_result.get('overall_inconsistency', 0)
    inconsistency_scores = face_result.get('inconsistency_scores', {})
    metadata_score = metadata_result.get('tampering_score', 0)
    tampering_signals = metadata_result.get('tampering_signals', [])

    findings = []
    severity_map = {}

    # ── Ensemble AI Signal ──────────────────────────────────────────────────
    model_scores = ensemble_result.get('per_model_scores', {})
    if model_scores:
        highest_model = max(model_scores, key=model_scores.get)
        highest_val = model_scores[highest_model]
        if is_fake:
            findings.append(
                f"🤖 **Ensemble AI:** {highest_model} shows the strongest signal "
                f"({highest_val:.1f}% fake probability), indicating AI-generated facial features."
            )
        else:
            findings.append(
                f"✅ **Ensemble AI:** All three models ({', '.join(model_scores.keys())}) "
                f"agree this content is authentic with {confidence:.1f}% confidence."
            )
        severity_map['ensemble'] = 'high' if fake_pct > 70 else 'medium' if fake_pct > 40 else 'low'

    # ── Face Inconsistency Signals ──────────────────────────────────────────
    if face_detected:
        st = inconsistency_scores.get('skin_texture', 0)
        ba = inconsistency_scores.get('boundary_artifacts', 0)
        li = inconsistency_scores.get('lighting_inconsistency', 0)
        ea = inconsistency_scores.get('eye_anomaly', 0)
        fa = inconsistency_scores.get('frequency_artifacts', 0)

        if st > 55:
            findings.append(
                f"🔬 **Skin Texture:** Abnormally smooth skin texture detected (score: {st:.0f}/100). "
                f"GAN models often over-smooth skin, removing natural pores and micro-expressions."
            )
        if ba > 50:
            findings.append(
                f"⚡ **Boundary Artifacts:** Sharp statistical discontinuity at face boundaries "
                f"(score: {ba:.0f}/100) — classic indicator of face-swap compositing."
            )
        if li > 55:
            findings.append(
                f"💡 **Lighting Inconsistency:** Facial illumination asymmetry score {li:.0f}/100 "
                f"suggests the face was composited under different lighting conditions."
            )
        if ea > 60:
            findings.append(
                f"👁️ **Eye Region Anomaly:** Eye clarity score {ea:.0f}/100 — "
                f"GAN models frequently fail to reproduce natural eye reflections and blinking patterns."
            )
        if fa > 50:
            findings.append(
                f"📡 **Frequency Fingerprint:** High-frequency artifact pattern score {fa:.0f}/100 — "
                f"GAN generators leave characteristic periodic noise invisible to the human eye."
            )

        if inconsistency < 25 and not is_fake:
            findings.append(
                f"✅ **Face Analysis:** All facial consistency checks pass "
                f"(overall inconsistency: {inconsistency:.0f}/100). Natural face characteristics confirmed."
            )
    else:
        findings.append(
            "⚠️ **Face Detection:** No face detected. Analysis performed on full image. "
            "Accuracy may be reduced for non-face content."
        )

    # ── Metadata Signals ────────────────────────────────────────────────────
    if metadata_score > 0:
        for signal in tampering_signals[:3]:
            findings.append(f"🗂️ **Metadata:** {signal}")
    if metadata_score == 0 and metadata_result.get('exif_present'):
        findings.append(
            "✅ **Metadata:** Complete camera EXIF data present with consistent timestamps and camera model."
        )

    # ── Verdict Summary ─────────────────────────────────────────────────────
    if is_fake:
        if confidence > 85:
            verdict = (
                f"VeriFace AI has detected this content as a **DEEPFAKE** with {confidence:.1f}% confidence. "
                f"Multiple independent analysis layers — neural ensemble, facial geometry, frequency analysis, "
                f"and metadata forensics — all converge on manipulation. Do not trust this content."
            )
        elif confidence > 65:
            verdict = (
                f"This content shows **strong indicators of AI manipulation** ({confidence:.1f}% confidence). "
                f"Several anomalies were detected in facial structure and frequency patterns. "
                f"Exercise significant caution."
            )
        else:
            verdict = (
                f"This content is **likely manipulated** ({confidence:.1f}% confidence), though some signals "
                f"are inconclusive. Recommend further manual review."
            )
    else:
        if confidence > 85:
            verdict = (
                f"VeriFace AI has verified this content as **AUTHENTIC** with {confidence:.1f}% confidence. "
                f"No deepfake artifacts, metadata tampering, or facial inconsistencies were detected."
            )
        elif confidence > 65:
            verdict = (
                f"This content appears **authentic** ({confidence:.1f}% confidence). "
                f"No significant manipulation signals detected, though confidence is moderate."
            )
        else:
            verdict = (
                f"Analysis is **inconclusive** ({confidence:.1f}% confidence). "
                f"Results lean toward authentic, but low confidence warrants manual review."
            )

    recommendation = (
        "🚨 Do not share, distribute, or use this content without verification from the original source."
        if is_fake else
        "✅ Content appears authentic. Standard digital media hygiene practices still recommended."
    )

    return {
        'verdict_summary': verdict,
        'key_findings': findings,
        'recommendation': recommendation,
        'total_signals_analyzed': len(findings),
        'risk_level': 'HIGH' if fake_pct > 65 else 'MEDIUM' if fake_pct > 35 else 'LOW',
    }
