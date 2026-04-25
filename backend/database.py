"""
VeriFace AI — SQLAlchemy Database Layer
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./verifaceai.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class ScanModel(Base):
    __tablename__ = "scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    is_fake = Column(Boolean, nullable=False)
    label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    deepfake_percentage = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    inference_time_ms = Column(Float, default=0)
    per_model_scores = Column(Text, default="{}")
    face_detected = Column(Boolean, default=False)
    metadata_score = Column(Integer, default=0)
    explanation_summary = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_scan(db: Session, scan_data: dict) -> ScanModel:
    """Save scan result to database."""
    scan = ScanModel(
        user_id=scan_data.get('user_id'),
        filename=scan_data.get('filename', 'unknown'),
        file_type=scan_data.get('file_type', 'image'),
        is_fake=scan_data.get('is_fake', False),
        label=scan_data.get('label', 'UNKNOWN'),
        confidence=scan_data.get('confidence', 0),
        deepfake_percentage=scan_data.get('deepfake_percentage', 0),
        risk_level=scan_data.get('risk_level', 'LOW'),
        inference_time_ms=scan_data.get('inference_time_ms', 0),
        per_model_scores=json.dumps(scan_data.get('per_model_scores', {})),
        face_detected=scan_data.get('face_detected', False),
        metadata_score=scan_data.get('metadata_score', 0),
        explanation_summary=scan_data.get('explanation_summary', ''),
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


def get_recent_scans(db: Session, limit: int = 50) -> list:
    return db.query(ScanModel).order_by(ScanModel.created_at.desc()).limit(limit).all()


def get_scan_by_id(db: Session, scan_id: int) -> ScanModel:
    return db.query(ScanModel).filter(ScanModel.id == scan_id).first()
