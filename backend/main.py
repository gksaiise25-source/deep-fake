"""
VeriFace AI — FastAPI Main Application
Run: uvicorn backend.main:app --reload --port 8000
"""

import os
import json
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

# Load env
from dotenv import load_dotenv
load_dotenv()

from backend.database import create_tables, get_db, save_scan, get_recent_scans, get_scan_by_id
from core.ensemble import get_detector
from core.face_analyzer import get_face_analyzer
from core.gradcam import generate_gradcam_heatmap
from core.metadata_checker import analyze_metadata
from core.explainer import generate_explanation
from utils.preprocessing import load_image_rgb, save_temp_file
from utils.pdf_report import generate_pdf_report

logger = logging.getLogger(__name__)

# Init DB tables on startup
create_tables()

app = FastAPI(
    title="VeriFace AI",
    description="🛡️ Advanced Deepfake Detection API — Ensemble AI + Cyber Forensics",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "VeriFace AI",
        "version": "2.0.0",
        "status": "operational",
        "models": ["EfficientNetB4", "Xception", "ViT-B/16"],
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/analyze", tags=["Detection"])
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Analyze an image or video for deepfake content.
    Returns full ensemble analysis, face inspection, metadata forensics, and AI explanation.
    """
    allowed_types = {
        "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp",
        "video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"
    }

    content_type = file.content_type or "application/octet-stream"
    is_video = content_type.startswith("video/")
    suffix = Path(file.filename or "upload").suffix or (".mp4" if is_video else ".jpg")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    tmp_path = save_temp_file(contents, suffix)

    try:
        detector = get_detector()
        face_analyzer = get_face_analyzer()

        if is_video:
            from core.video_analyzer import VideoAnalyzer
            va = VideoAnalyzer(ensemble_detector=detector, face_analyzer=face_analyzer)
            video_result = va.analyze_video(tmp_path)
            ensemble_result = {
                'is_fake': video_result['is_fake'],
                'label': video_result['label'],
                'fake_probability': video_result['fake_probability'],
                'confidence': video_result['confidence'],
                'deepfake_percentage': video_result['deepfake_percentage'],
                'per_model_scores': {'Ensemble': video_result['deepfake_percentage']},
                'inference_time_ms': 0,
                'is_demo_mode': True,
            }
            face_result = {'face_detected': False, 'overall_inconsistency': 0, 'inconsistency_scores': {}}
        else:
            image = load_image_rgb(tmp_path)
            if image is None:
                raise HTTPException(status_code=422, detail="Cannot decode image")
            ensemble_result = detector.analyze(image)
            faces = face_analyzer.detect_faces(image)
            face_result = face_analyzer.analyze_inconsistencies(image, faces)

        metadata_result = analyze_metadata(tmp_path)
        explanation = generate_explanation(ensemble_result, face_result, metadata_result)

        # Save to DB
        scan = save_scan(db, {
            'filename': file.filename,
            'file_type': 'video' if is_video else 'image',
            'is_fake': ensemble_result['is_fake'],
            'label': ensemble_result['label'],
            'confidence': ensemble_result['confidence'],
            'deepfake_percentage': ensemble_result['deepfake_percentage'],
            'risk_level': explanation.get('risk_level', 'LOW'),
            'inference_time_ms': ensemble_result.get('inference_time_ms', 0),
            'per_model_scores': ensemble_result.get('per_model_scores', {}),
            'face_detected': face_result.get('face_detected', False),
            'metadata_score': metadata_result.get('tampering_score', 0),
            'explanation_summary': explanation.get('verdict_summary', ''),
        })

        return JSONResponse({
            "success": True,
            "scan_id": scan.id,
            "filename": file.filename,
            "file_type": "video" if is_video else "image",
            "ensemble": ensemble_result,
            "face_analysis": face_result,
            "metadata": metadata_result,
            "explanation": explanation,
            "created_at": scan.created_at.isoformat(),
        })

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/history", tags=["History"])
async def get_history(limit: int = 50, db: Session = Depends(get_db)):
    """Get recent scan history."""
    scans = get_recent_scans(db, limit=limit)
    return [{
        "id": s.id,
        "filename": s.filename,
        "file_type": s.file_type,
        "label": s.label,
        "confidence": s.confidence,
        "risk_level": s.risk_level,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    } for s in scans]


@app.get("/api/report/{scan_id}", tags=["Reports"])
async def download_report(scan_id: int, db: Session = Depends(get_db)):
    """Download PDF forensic report for a scan."""
    scan = get_scan_by_id(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    per_model = json.loads(scan.per_model_scores) if scan.per_model_scores else {}
    ensemble_result = {
        'is_fake': scan.is_fake, 'label': scan.label,
        'confidence': scan.confidence, 'deepfake_percentage': scan.deepfake_percentage,
        'per_model_scores': per_model,
    }
    face_result = {'face_detected': scan.face_detected, 'overall_inconsistency': 0, 'inconsistency_scores': {}}
    metadata_result = {'file_name': scan.filename, 'file_size_kb': 0, 'tampering_score': scan.metadata_score}
    explanation = {
        'verdict_summary': scan.explanation_summary,
        'key_findings': [],
        'recommendation': '',
        'risk_level': scan.risk_level
    }

    pdf_bytes = generate_pdf_report(ensemble_result, face_result, metadata_result, explanation)
    if not pdf_bytes:
        raise HTTPException(status_code=503, detail="PDF generation unavailable")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=verifaceai_report_{scan_id}.pdf"}
    )


@app.delete("/api/history/{scan_id}", tags=["History"])
async def delete_scan(scan_id: int, db: Session = Depends(get_db)):
    scan = get_scan_by_id(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    db.delete(scan)
    db.commit()
    return {"success": True, "message": f"Scan {scan_id} deleted"}
