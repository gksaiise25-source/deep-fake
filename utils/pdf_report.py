"""
VeriFace AI — PDF Forensic Report Generator
Generates a professional branded PDF forensic report after each analysis.
"""

import os
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    logger.warning("reportlab not available — PDF reports disabled")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Brand Colors
BRAND_DARK = colors.HexColor("#0a0a0f")
BRAND_CYAN = colors.HexColor("#00f5ff")
BRAND_PURPLE = colors.HexColor("#7c3aed")
BRAND_RED = colors.HexColor("#ef4444")
BRAND_GREEN = colors.HexColor("#22c55e")
BRAND_GRAY = colors.HexColor("#94a3b8")
BRAND_LIGHT_BG = colors.HexColor("#f8fafc")
BRAND_CARD = colors.HexColor("#1e293b")


def generate_pdf_report(
    ensemble_result: Dict,
    face_result: Dict,
    metadata_result: Dict,
    explanation: Dict,
    image_array: Optional[np.ndarray] = None,
    heatmap_array: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> Optional[bytes]:
    """
    Generate a full forensic PDF report.

    Returns:
        PDF bytes if successful, None otherwise
    """
    if not REPORTLAB_AVAILABLE:
        logger.warning("PDF generation skipped — reportlab not installed")
        return None

    buf = io.BytesIO()
    target = output_path or buf

    doc = SimpleDocTemplate(
        target if isinstance(target, str) else buf,
        pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=20*mm
    )

    styles = getSampleStyleSheet()
    story = []

    # ─── Custom Styles ─────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        'Title', parent=styles['Normal'],
        fontSize=24, fontName='Helvetica-Bold',
        textColor=BRAND_CYAN, alignment=TA_CENTER, spaceAfter=2*mm
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=11, fontName='Helvetica',
        textColor=BRAND_GRAY, alignment=TA_CENTER, spaceAfter=6*mm
    )
    section_style = ParagraphStyle(
        'Section', parent=styles['Normal'],
        fontSize=13, fontName='Helvetica-Bold',
        textColor=BRAND_PURPLE, spaceBefore=5*mm, spaceAfter=3*mm
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=9, fontName='Helvetica',
        textColor=colors.HexColor("#334155"), spaceAfter=2*mm, leading=14
    )
    verdict_fake_style = ParagraphStyle(
        'VerdictFake', parent=styles['Normal'],
        fontSize=18, fontName='Helvetica-Bold',
        textColor=BRAND_RED, alignment=TA_CENTER, spaceAfter=3*mm
    )
    verdict_real_style = ParagraphStyle(
        'VerdictReal', parent=styles['Normal'],
        fontSize=18, fontName='Helvetica-Bold',
        textColor=BRAND_GREEN, alignment=TA_CENTER, spaceAfter=3*mm
    )

    # ─── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("🛡️ VeriFace AI", title_style))
    story.append(Paragraph("Cyber Forensic Deepfake Detection Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_CYAN))
    story.append(Spacer(1, 5*mm))

    # ─── Report Metadata ───────────────────────────────────────────────────
    now = datetime.now()
    meta_data = [
        ["Report Generated", now.strftime("%B %d, %Y at %H:%M:%S")],
        ["Analysis Engine", "VeriFace AI v2.0 — Ensemble (EfficientNetB4 + Xception + ViT-B/16)"],
        ["File Analyzed", metadata_result.get('file_name', 'Unknown')],
        ["File Size", f"{metadata_result.get('file_size_kb', 0):.1f} KB"],
        ["Report ID", f"VFA-{now.strftime('%Y%m%d%H%M%S')}"],
    ]
    meta_table = Table(meta_data, colWidths=[50*mm, 125*mm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), BRAND_PURPLE),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor("#334155")),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5*mm))

    # ─── Verdict Banner ────────────────────────────────────────────────────
    is_fake = ensemble_result.get('is_fake', False)
    confidence = ensemble_result.get('confidence', 0)
    label = "⚠️  DEEPFAKE DETECTED" if is_fake else "✅  CONTENT AUTHENTICATED"
    verdict_style = verdict_fake_style if is_fake else verdict_real_style
    banner_color = colors.HexColor("#fef2f2") if is_fake else colors.HexColor("#f0fdf4")
    border_color = BRAND_RED if is_fake else BRAND_GREEN

    verdict_table = Table(
        [[Paragraph(label, verdict_style)],
         [Paragraph(f"Confidence: {confidence:.1f}%  |  Risk Level: {explanation.get('risk_level', 'N/A')}", subtitle_style)]],
        colWidths=[175*mm]
    )
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_color),
        ('BOX', (0, 0), (-1, -1), 2, border_color),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 4*mm))

    # ─── Ensemble Scores ───────────────────────────────────────────────────
    story.append(Paragraph("Neural Network Ensemble Scores", section_style))
    model_scores = ensemble_result.get('per_model_scores', {})
    if model_scores:
        score_data = [["Model", "Deepfake Score", "Verdict"]]
        for model_name, score in model_scores.items():
            v = "FAKE" if score > 50 else "REAL"
            vc = BRAND_RED if score > 50 else BRAND_GREEN
            score_data.append([model_name, f"{score:.1f}%", v])
        score_data.append([
            "ENSEMBLE (Weighted)",
            f"{ensemble_result.get('deepfake_percentage', 0):.1f}%",
            ensemble_result.get('label', '')
        ])
        score_table = Table(score_data, colWidths=[70*mm, 55*mm, 50*mm])
        score_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), BRAND_PURPLE),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#f8fafc")]),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#f1f5f9")),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(score_table)
    story.append(Spacer(1, 3*mm))

    # ─── Face Analysis ─────────────────────────────────────────────────────
    story.append(Paragraph("Facial Inconsistency Analysis", section_style))
    incon_scores = face_result.get('inconsistency_scores', {})
    if incon_scores:
        face_data = [["Signal", "Score / 100", "Risk"]]
        labels = {
            'skin_texture': 'Skin Texture Smoothness',
            'boundary_artifacts': 'Boundary Artifacts',
            'lighting_inconsistency': 'Lighting Inconsistency',
            'eye_anomaly': 'Eye Region Anomaly',
            'color_inconsistency': 'Color Channel Inconsistency',
            'frequency_artifacts': 'Frequency Domain Fingerprint'
        }
        for key, display in labels.items():
            val = incon_scores.get(key, 0)
            risk = "HIGH" if val > 65 else "MED" if val > 35 else "LOW"
            face_data.append([display, f"{val:.0f}", risk])
        face_table = Table(face_data, colWidths=[90*mm, 45*mm, 40*mm])
        face_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), BRAND_PURPLE),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(face_table)
    story.append(Spacer(1, 3*mm))

    # ─── Metadata Analysis ─────────────────────────────────────────────────
    story.append(Paragraph("Metadata Forensics", section_style))
    meta_analysis = [
        ["EXIF Present", "Yes" if metadata_result.get('exif_present') else "No"],
        ["Camera Info", metadata_result.get('camera_info') or "Not found"],
        ["Creation Date", metadata_result.get('creation_date') or "Not found"],
        ["GAN Signature", "YES — SUSPICIOUS" if metadata_result.get('gan_signature_found') else "Not found"],
        ["Tampering Verdict", metadata_result.get('tampering_verdict', 'N/A')],
        ["Tampering Score", f"{metadata_result.get('tampering_score', 0)}/100"],
    ]
    m_table = Table(meta_analysis, colWidths=[60*mm, 115*mm])
    m_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), BRAND_PURPLE),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 3*mm))

    # ─── AI Explanation ────────────────────────────────────────────────────
    story.append(Paragraph("AI Forensic Findings", section_style))
    story.append(Paragraph(explanation.get('verdict_summary', ''), body_style))
    story.append(Spacer(1, 2*mm))

    for finding in explanation.get('key_findings', []):
        clean = finding.replace('**', '').replace('*', '')
        story.append(Paragraph(f"• {clean}", body_style))

    story.append(Spacer(1, 3*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_GRAY))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(explanation.get('recommendation', ''), body_style))

    # ─── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_CYAN))
    story.append(Paragraph(
        "VeriFace AI — Advanced Deepfake Detection System | Cyber Forensics Division | "
        f"Report generated {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7,
                       textColor=BRAND_GRAY, alignment=TA_CENTER)
    ))

    doc.build(story)

    if output_path:
        return None
    return buf.getvalue()
