"""
main.py  –  FastAPI Application
--------------------------------
REST API endpoints for all ML modules:
  POST /api/v1/claims/predict       → Claim approval prediction
  POST /api/v1/fraud/detect         → Fraud detection
  POST /api/v1/nlp/analyze          → Medical text NLP
  POST /api/v1/recommend            → Plan recommendations
  GET  /api/v1/health               → Health check
  GET  /api/v1/models/status        → Model registry status
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Health Insurance AI Platform – Lumen Technologies",
    description="AI/ML APIs for claim approval, fraud detection, NLP, and recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─── Pydantic Schemas ──────────────────────────────────────────────────────────


class ClaimRequest(BaseModel):
    member_id: str
    claim_amount: float = Field(..., gt=0)
    claim_type: str = Field(..., example="Outpatient")
    diagnosis_code: str = Field(..., example="I10")
    procedure_code: str = Field(..., example="99213")
    num_procedures: int = Field(1, ge=1)
    days_in_hospital: int = Field(0, ge=0)
    prior_auth: int = Field(0, ge=0, le=1)
    member_age: int = Field(..., ge=18, le=100)
    member_bmi: float = Field(..., ge=10, le=70)
    member_smoker: int = Field(0, ge=0, le=1)
    member_plan: str = Field("Silver", example="Silver")
    num_chronic: int = Field(0, ge=0)
    state: str = Field("CA", example="CA")


class ClaimResponse(BaseModel):
    claim_id: str
    member_id: str
    approval_probability: float
    risk_score: float
    risk_band: str
    decision: str
    processing_time_ms: float


class FraudRequest(BaseModel):
    claim_id: str
    member_id: str
    provider_id: str
    claim_amount: float
    num_procedures: int
    days_in_hospital: int
    provider_avg_claim: Optional[float] = 5000.0
    provider_claim_count: Optional[int] = 100
    provider_unique_members: Optional[int] = 80
    member_claim_count: Optional[int] = 5
    member_avg_claim: Optional[float] = 3000.0
    member_total_spend: Optional[float] = 15000.0


class FraudResponse(BaseModel):
    claim_id: str
    is_anomaly: int
    anomaly_score: float
    fraud_flag: int
    high_fraud_cluster: int
    combined_fraud_score: float
    alert_level: str


class NLPRequest(BaseModel):
    note_id: Optional[str] = "NOTE000001"
    clinical_note: str = Field(..., min_length=10)


class NLPResponse(BaseModel):
    note_id: str
    cleaned_text: str
    predicted_diagnosis: str
    confidence: float
    sentiment: Optional[str] = None
    processing_time_ms: float


class RecommendRequest(BaseModel):
    member_id: str
    age: int = Field(..., ge=18, le=100)
    bmi: float = Field(..., ge=10, le=70)
    num_chronic_conditions: int = Field(0, ge=0)
    tenure_months: int = Field(12, ge=1)
    annual_premium: float = Field(5000.0, gt=0)
    deductible: int = Field(1000, gt=0)
    income_score: float = Field(0.5, ge=0, le=1)


class RecommendResponse(BaseModel):
    member_id: str
    recommendations: List[dict]
    explanation: str


# ─── Model Cache ──────────────────────────────────────────────────────────────
_models = {}


def get_claim_model():
    if "claim" not in _models:
        try:
            from src.models.claim_approval_model import ClaimApprovalModel

            _models["claim"] = ClaimApprovalModel.load("xgboost")
            logger.info("✓ Claim model loaded")
        except Exception as e:
            logger.warning(f"Claim model not found: {e}. Using mock.")
            _models["claim"] = None
    return _models["claim"]


def get_fraud_model():
    if "fraud" not in _models:
        try:
            from src.models.fraud_detection_model import FraudDetectionModel

            _models["fraud"] = FraudDetectionModel.load()
            logger.info("✓ Fraud model loaded")
        except Exception as e:
            logger.warning(f"Fraud model not found: {e}. Using mock.")
            _models["fraud"] = None
    return _models["fraud"]


def get_recommender():
    if "recommender" not in _models:
        try:
            from src.models.recommendation_engine import HybridRecommender

            _models["recommender"] = HybridRecommender.load()
            logger.info("✓ Recommender loaded")
        except Exception as e:
            logger.warning(f"Recommender not found: {e}. Using mock.")
            _models["recommender"] = None
    return _models["recommender"]


# ─── Mock Predictions (fallback when models not trained) ──────────────────────
def _mock_claim_predict(req: ClaimRequest) -> dict:
    import random, hashlib

    seed = int(hashlib.md5(req.member_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    prob = rng.uniform(0.5, 0.98)
    risk = 1 - prob
    bands = ["Very Low", "Low", "Medium", "High", "Very High"]
    band = bands[min(int(risk * 5), 4)]
    return {
        "approval_probability": round(prob, 4),
        "risk_score": round(risk, 4),
        "risk_band": band,
        "decision": "Approve" if prob >= 0.5 else "Review",
    }


def _mock_fraud_predict(req: FraudRequest) -> dict:
    import random, hashlib

    seed = int(hashlib.md5(req.claim_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    score = rng.uniform(0.0, 0.4)
    fraud = 1 if score > 0.3 else 0
    return {
        "is_anomaly": fraud,
        "anomaly_score": round(score, 4),
        "fraud_flag": fraud,
        "high_fraud_cluster": fraud,
        "combined_fraud_score": round(score, 4),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/api/v1/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "Health Insurance AI Platform",
        "version": "1.0.0",
        "models_loaded": list(_models.keys()),
    }


@app.get("/api/v1/models/status", tags=["System"])
def models_status():
    return {
        "claim_approval": "loaded" if _models.get("claim") else "not_loaded",
        "fraud_detection": "loaded" if _models.get("fraud") else "not_loaded",
        "recommender": "loaded" if _models.get("recommender") else "not_loaded",
        "nlp": "loaded" if _models.get("nlp") else "not_loaded",
    }


@app.post("/api/v1/claims/predict", response_model=ClaimResponse, tags=["Claims"])
def predict_claim(req: ClaimRequest, background_tasks: BackgroundTasks):
    """
    Predict claim approval probability and risk score.
    Returns decision: Approve or Review.
    """
    t0 = time.time()
    model = get_claim_model()
    import uuid
    import time as _time

    if model:
        from src.preprocessing.feature_engineering import ClaimFeatureEngineer

        engineer = ClaimFeatureEngineer.load("claim_feature_engineer")
        df = pd.DataFrame([req.dict()])
        X = engineer.transform(df)
        result = model.risk_score(X).iloc[0].to_dict()
    else:
        result = _mock_claim_predict(req)

    return ClaimResponse(
        claim_id=f"CLM{uuid.uuid4().hex[:8].upper()}",
        member_id=req.member_id,
        approval_probability=result["approval_probability"],
        risk_score=result["risk_score"],
        risk_band=result["risk_band"],
        decision=result["decision"],
        processing_time_ms=0.0,
    )


@app.post("/api/v1/fraud/detect", response_model=FraudResponse, tags=["Fraud"])
def detect_fraud(req: FraudRequest):
    """
    Detect potential fraud using Isolation Forest + K-Means ensemble.
    Returns anomaly score and fraud flag.
    """
    t0 = time.time()
    model = get_fraud_model()

    if model:
        from src.preprocessing.feature_engineering import FraudFeatureEngineer

        engineer = FraudFeatureEngineer.load("fraud_feature_engineer")
        df = pd.DataFrame([req.dict()])
        X = engineer.transform(df)
        result = model.predict_combined(X).iloc[0].to_dict()
    else:
        result = _mock_fraud_predict(req)

    score = result["combined_fraud_score"]
    alert = "HIGH" if score > 0.75 else "MEDIUM" if score > 0.5 else "LOW"

    return FraudResponse(
        claim_id=req.claim_id,
        is_anomaly=int(result["is_anomaly"]),
        anomaly_score=float(result["anomaly_score"]),
        fraud_flag=int(result["fraud_flag"]),
        high_fraud_cluster=int(result["high_fraud_cluster"]),
        combined_fraud_score=float(result["combined_fraud_score"]),
        alert_level=alert,
    )


@app.post("/api/v1/nlp/analyze", response_model=NLPResponse, tags=["NLP"])
def analyze_medical_text(req: NLPRequest):
    """
    Analyze clinical notes: clean text, predict diagnosis, analyze sentiment.
    """
    t0 = time.time()
    from src.models.nlp_medical_text import MedicalTextPreprocessor

    preprocessor = MedicalTextPreprocessor()
    cleaned = preprocessor.clean(req.clinical_note)

    # Attempt BERT prediction, fallback to keyword heuristic
    try:
        from src.models.nlp_medical_text import BERTDiagnosisClassifier

        bert = BERTDiagnosisClassifier()
        preds = bert.predict([cleaned])
        diagnosis = preds.iloc[0]["predicted_diagnosis"]
        confidence = float(preds.iloc[0]["confidence"])
    except Exception:
        # Keyword fallback
        diagnosis_map = {
            "hypertension": ("I10", 0.85),
            "diabetes": ("E11", 0.82),
            "asthma": ("J45", 0.80),
            "pain": ("M54", 0.70),
            "cancer": ("C34", 0.75),
            "depression": ("F32", 0.78),
        }
        diagnosis, confidence = "J06", 0.60
        for kw, (dx, conf) in diagnosis_map.items():
            if kw in cleaned:
                diagnosis, confidence = dx, conf
                break

    return NLPResponse(
        note_id=req.note_id,
        cleaned_text=cleaned[:200],
        predicted_diagnosis=diagnosis,
        confidence=confidence,
        sentiment="neutral",
        processing_time_ms=0.0,
    )


@app.post(
    "/api/v1/recommend", response_model=RecommendResponse, tags=["Recommendations"]
)
def recommend_plan(req: RecommendRequest):
    """
    Recommend personalized insurance plans using Hybrid CF + KNN + Content-Based.
    """
    recommender = get_recommender()
    profile = req.dict()

    if recommender:
        recs = recommender.recommend(req.member_id, profile, top_n=3)
    else:
        # Heuristic fallback
        age, chronic = req.age, req.num_chronic_conditions
        if age > 60 or chronic >= 3:
            recs = [
                {"plan": "Platinum", "score": 0.90},
                {"plan": "Gold", "score": 0.75},
                {"plan": "Silver", "score": 0.55},
            ]
        elif age > 40 or chronic >= 1:
            recs = [
                {"plan": "Gold", "score": 0.85},
                {"plan": "Silver", "score": 0.72},
                {"plan": "Platinum", "score": 0.60},
            ]
        else:
            recs = [
                {"plan": "Silver", "score": 0.88},
                {"plan": "Bronze", "score": 0.70},
                {"plan": "Gold", "score": 0.55},
            ]

    top_plan = recs[0]["plan"] if recs else "Silver"
    explanation = (
        f"Based on your profile (age {req.age}, {req.num_chronic_conditions} chronic conditions), "
        f"the {top_plan} plan offers the best balance of coverage and cost."
    )

    return RecommendResponse(
        member_id=req.member_id,
        recommendations=recs,
        explanation=explanation,
    )


@app.post("/api/v1/claims/batch", tags=["Claims"])
def predict_claims_batch(claims: List[ClaimRequest], background_tasks: BackgroundTasks):
    """Batch claim prediction — processes up to 100 claims per request."""
    if len(claims) > 100:
        raise HTTPException(status_code=400, detail="Max 100 claims per batch request")
    results = [predict_claim(c, background_tasks) for c in claims]
    return {
        "total": len(results),
        "approved": sum(1 for r in results if r.decision == "Approve"),
        "review": sum(1 for r in results if r.decision == "Review"),
        "results": results,
    }


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Health Insurance AI Platform starting...")
    get_claim_model()
    get_fraud_model()
    get_recommender()
    logger.info("✅ API ready — visit /docs for Swagger UI")
