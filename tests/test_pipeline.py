"""
test_pipeline.py
----------------
Unit tests for all modules: ingestion, preprocessing, models, API
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_data():
    """Generate a small synthetic dataset for all tests."""
    from src.ingestion.generate_synthetic_data import generate_members, generate_claims, generate_providers
    members   = generate_members(n=500)
    claims    = generate_claims(members, n=2000)
    providers = generate_providers(n=50)
    return {"members": members, "claims": claims, "providers": providers}


@pytest.fixture(scope="session")
def processed_data(synthetic_data):
    from src.preprocessing.feature_engineering import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    return pipeline.fit_transform_all(synthetic_data)


# ─── Ingestion Tests ─────────────────────────────────────────────────────────

class TestDataIngestion:
    def test_members_shape(self, synthetic_data):
        df = synthetic_data["members"]
        assert len(df) == 500
        assert "member_id" in df.columns

    def test_claims_shape(self, synthetic_data):
        df = synthetic_data["claims"]
        assert len(df) == 2000
        assert "claim_id" in df.columns

    def test_fraud_rate_reasonable(self, synthetic_data):
        fraud_rate = synthetic_data["claims"]["is_fraud"].mean()
        assert 0.01 < fraud_rate < 0.15, f"Unexpected fraud rate: {fraud_rate}"

    def test_no_duplicate_claim_ids(self, synthetic_data):
        claims = synthetic_data["claims"]
        assert claims["claim_id"].nunique() == len(claims)

    def test_claim_amounts_positive(self, synthetic_data):
        assert (synthetic_data["claims"]["claim_amount"] > 0).all()


# ─── Preprocessing Tests ──────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_claim_features_shape(self, processed_data):
        X = processed_data["claim_features"]
        assert X.shape[0] > 0
        assert X.shape[1] > 5

    def test_no_nulls_in_features(self, processed_data):
        X = processed_data["claim_features"]
        assert not X.isnull().any().any(), "Feature matrix contains NaN values"

    def test_fraud_features_shape(self, processed_data):
        X = processed_data["fraud_features"]
        assert X.shape[0] > 0

    def test_labels_binary(self, processed_data):
        y = processed_data["claim_labels"]
        assert set(y.unique()).issubset({0, 1})


# ─── Claim Approval Model Tests ───────────────────────────────────────────────

class TestClaimApprovalModel:
    def test_model_trains(self, processed_data):
        from src.models.claim_approval_model import ClaimApprovalModel
        model = ClaimApprovalModel(model_type="logistic")
        metrics = model.train(
            processed_data["claim_features"],
            processed_data["claim_labels"]
        )
        assert "auc_roc" in metrics
        assert metrics["auc_roc"] > 0.5

    def test_predict_returns_correct_shape(self, processed_data):
        from src.models.claim_approval_model import ClaimApprovalModel
        model = ClaimApprovalModel(model_type="logistic")
        model.train(processed_data["claim_features"], processed_data["claim_labels"])
        preds = model.predict_proba(processed_data["claim_features"].head(10))
        assert len(preds) == 10
        assert all(0 <= p <= 1 for p in preds)

    def test_risk_score_output(self, processed_data):
        from src.models.claim_approval_model import ClaimApprovalModel
        model = ClaimApprovalModel(model_type="logistic")
        model.train(processed_data["claim_features"], processed_data["claim_labels"])
        risk_df = model.risk_score(processed_data["claim_features"].head(5))
        assert "approval_probability" in risk_df.columns
        assert "decision" in risk_df.columns
        assert set(risk_df["decision"]).issubset({"Approve", "Review"})


# ─── Fraud Detection Tests ────────────────────────────────────────────────────

class TestFraudDetection:
    def test_isolation_forest_trains(self, processed_data):
        from src.models.fraud_detection_model import FraudDetectionModel
        model = FraudDetectionModel()
        result = model.train_isolation_forest(processed_data["fraud_features"])
        assert "anomaly_rate" in result
        assert 0 < result["anomaly_rate"] < 0.5

    def test_kmeans_trains(self, processed_data):
        from src.models.fraud_detection_model import FraudDetectionModel
        model = FraudDetectionModel()
        model.train_isolation_forest(processed_data["fraud_features"])
        result = model.train_kmeans(processed_data["fraud_features"],
                                     processed_data["fraud_labels"])
        assert len(result["clusters"]) == 5

    def test_combined_predict(self, processed_data):
        from src.models.fraud_detection_model import FraudDetectionModel
        model = FraudDetectionModel()
        model.train_isolation_forest(processed_data["fraud_features"])
        model.train_kmeans(processed_data["fraud_features"], processed_data["fraud_labels"])
        preds = model.predict_combined(processed_data["fraud_features"].head(20))
        assert "combined_fraud_score" in preds.columns
        assert "final_fraud_flag" in preds.columns
        assert all(0 <= s <= 1 for s in preds["combined_fraud_score"])


# ─── NLP Tests ───────────────────────────────────────────────────────────────

class TestNLP:
    def test_text_cleaning(self):
        from src.models.nlp_medical_text import MedicalTextPreprocessor
        proc = MedicalTextPreprocessor()
        raw  = "Pt presents w/ HTN. BP 145/90. Dx: DM type 2."
        clean = proc.clean(raw)
        assert "patient" in clean
        assert "hypertension" in clean
        assert clean == clean.lower()

    def test_batch_cleaning(self):
        from src.models.nlp_medical_text import MedicalTextPreprocessor
        proc = MedicalTextPreprocessor()
        texts = pd.Series(["Pt has HTN", "Dx: DM", "sob with cp"])
        cleaned = proc.batch_clean(texts)
        assert len(cleaned) == 3
        assert all(isinstance(t, str) for t in cleaned)


# ─── Recommendation Tests ─────────────────────────────────────────────────────

class TestRecommendation:
    def test_content_based_recommender(self):
        from src.models.recommendation_engine import ContentBasedFilter
        cb = ContentBasedFilter()
        profile = {"age": 65, "bmi": 30, "num_chronic_conditions": 3,
                   "income_score": 0.3}
        recs = cb.recommend(profile, top_n=3)
        assert len(recs) == 3
        assert all("plan" in r and "score" in r for r in recs)
        # High-risk profile should prefer Gold/Platinum
        top_plan = recs[0]["plan"]
        assert top_plan in ["Gold", "Platinum"]

    def test_knn_recommender(self, synthetic_data):
        from src.models.recommendation_engine import KNNRecommender
        knn = KNNRecommender(k=5)
        knn.fit(synthetic_data["members"])
        profile = {"age": 40, "bmi": 25, "num_chronic_conditions": 1,
                   "tenure_months": 24, "annual_premium": 6000, "deductible": 1500}
        recs = knn.recommend(profile, top_n=3)
        assert len(recs) >= 1
        assert all("plan" in r for r in recs)


# ─── API Tests ────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_claim_predict_endpoint(self, client):
        payload = {
            "member_id": "MBR000001",
            "claim_amount": 5000.0,
            "claim_type": "Outpatient",
            "diagnosis_code": "I10",
            "procedure_code": "99213",
            "num_procedures": 2,
            "days_in_hospital": 0,
            "prior_auth": 1,
            "member_age": 45,
            "member_bmi": 27.5,
            "member_smoker": 0,
            "member_plan": "Silver",
            "num_chronic": 1,
            "state": "CA",
        }
        resp = client.post("/api/v1/claims/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "approval_probability" in data
        assert "decision" in data
        assert 0 <= data["approval_probability"] <= 1

    def test_fraud_detect_endpoint(self, client):
        payload = {
            "claim_id": "CLM0000001",
            "member_id": "MBR000001",
            "provider_id": "PRV0001",
            "claim_amount": 50000.0,
            "num_procedures": 15,
            "days_in_hospital": 0,
        }
        resp = client.post("/api/v1/fraud/detect", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "fraud_flag" in data
        assert "alert_level" in data
        assert data["alert_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_recommend_endpoint(self, client):
        payload = {
            "member_id": "MBR000001",
            "age": 55,
            "bmi": 28.0,
            "num_chronic_conditions": 2,
            "tenure_months": 36,
            "annual_premium": 8000.0,
            "deductible": 2000,
            "income_score": 0.5,
        }
        resp = client.post("/api/v1/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0

    def test_nlp_endpoint(self, client):
        payload = {
            "note_id": "NOTE000001",
            "clinical_note": "Patient presents with hypertension. BP 150/95. Prescribed Lisinopril.",
        }
        resp = client.post("/api/v1/nlp/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_diagnosis" in data
        assert "confidence" in data
