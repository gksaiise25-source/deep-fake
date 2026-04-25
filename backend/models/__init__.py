"""VeriFace AI — Pydantic Schemas"""

from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List, Any
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ScanResult(BaseModel):
    id: Optional[int] = None
    filename: str
    file_type: str
    is_fake: bool
    label: str
    confidence: float
    deepfake_percentage: float
    risk_level: str
    inference_time_ms: float
    per_model_scores: Dict[str, float]
    face_detected: bool
    metadata_score: int
    created_at: Optional[datetime] = None


class AnalyzeResponse(BaseModel):
    success: bool
    scan: ScanResult
    explanation: Dict[str, Any]
    face_analysis: Dict[str, Any]
    metadata_analysis: Dict[str, Any]
    message: str = "Analysis complete"
