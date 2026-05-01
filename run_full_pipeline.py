"""
run_full_pipeline.py
--------------------
Runs the complete ML pipeline locally.
Mimics the Airflow DAG without needing Airflow installed.

Steps:
  1. Ingest data
  2. Preprocess & feature engineering
  3. Train claim approval model
  4. Train fraud detection model
  5. Train recommendation engine
  6. Evaluate all models
  7. Start API

Usage: python run_full_pipeline.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)
Path("logs").mkdir(exist_ok=True)

def banner(step, title):
    print(f"\n{'='*60}")
    print(f"  STEP {step}: {title}")
    print(f"{'='*60}")

def main():
    start_time = time.time()
    results    = {}

    print("\n" + "🏥 " * 15)
    print("  HEALTH INSURANCE AI — FULL PIPELINE")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🏥 " * 15)

    # ── STEP 1: Generate Data ─────────────────────────────────
    banner(1, "DATA GENERATION")
    data_exists = Path("data/synthetic/claims.csv").exists()
    if data_exists:
        print("  Data already exists — skipping generation")
        print("  Delete data/synthetic/ to regenerate")
    else:
        print("  Generating synthetic datasets...")
        from src.ingestion.generate_synthetic_data import (
            generate_members, generate_claims,
            generate_providers, generate_clinical_notes
        )
        import pandas as pd
        Path("data/synthetic").mkdir(parents=True, exist_ok=True)
        members   = generate_members(10000)
        claims    = generate_claims(members, 50000)
        providers = generate_providers(500)
        notes     = generate_clinical_notes(5000)
        members.to_csv("data/synthetic/members.csv",       index=False)
        claims.to_csv("data/synthetic/claims.csv",         index=False)
        providers.to_csv("data/synthetic/providers.csv",   index=False)
        notes.to_csv("data/synthetic/clinical_notes.csv",  index=False)
        print(f"  Members   : {len(members):,}")
        print(f"  Claims    : {len(claims):,}  | fraud rate: {claims['is_fraud'].mean():.1%}")
        print(f"  Providers : {len(providers):,}")
        print(f"  Notes     : {len(notes):,}")

    # ── STEP 2: Ingest & Preprocess ───────────────────────────
    banner(2, "INGESTION & FEATURE ENGINEERING")
    from src.ingestion.data_ingestion import LocalIngestion
    from src.preprocessing.feature_engineering import PreprocessingPipeline

    print("  Loading data...")
    data = LocalIngestion().load_all()

    print("  Running feature engineering...")
    pipeline  = PreprocessingPipeline()
    processed = pipeline.fit_transform_all(data)

    print(f"  Claim features : {processed['claim_features'].shape}")
    print(f"  Fraud features : {processed['fraud_features'].shape}")
    print(f"  Approval rate  : {processed['claim_labels'].mean():.1%}")
    print(f"  Fraud rate     : {processed['fraud_labels'].mean():.1%}")

    # ── STEP 3: Train Claim Model ─────────────────────────────
    banner(3, "CLAIM APPROVAL MODEL")
    from src.models.claim_approval_model import ClaimApprovalModel

    best_auc   = 0
    best_model = "xgboost"

    for model_type in ["logistic", "xgboost", "lightgbm"]:
        print(f"\n  Training {model_type}...")
        model   = ClaimApprovalModel(model_type=model_type)
        metrics = model.train(
            processed["claim_features"],
            processed["claim_labels"]
        )
        print(f"  AUC: {metrics['auc_roc']} | CV: {metrics['cv_auc_mean']} ± {metrics['cv_auc_std']}")
        if metrics["auc_roc"] > best_auc:
            best_auc   = metrics["auc_roc"]
            best_model = model_type
        results[f"claim_{model_type}"] = metrics["auc_roc"]

    print(f"\n  Best claim model : {best_model} (AUC: {best_auc})")

    # ── STEP 4: Train Fraud Model ─────────────────────────────
    banner(4, "FRAUD DETECTION MODEL")
    from src.models.fraud_detection_model import FraudDetectionModel

    fraud_model = FraudDetectionModel()

    print("  Stage 1: Isolation Forest...")
    iso = fraud_model.train_isolation_forest(processed["fraud_features"])
    print(f"  Anomaly rate: {iso['anomaly_rate']:.1%}")

    print("  Stage 2: K-Means Clustering...")
    km = fraud_model.train_kmeans(
        processed["fraud_features"],
        processed["fraud_labels"]
    )
    print(f"  High-fraud clusters: {fraud_model.fraud_cluster_ids}")

    print("  Stage 3: Supervised XGBoost...")
    sup = fraud_model.train_supervised(
        processed["fraud_features"],
        processed["fraud_labels"]
    )
    print(f"  AUC-ROC: {sup['auc_roc']}")
    fraud_model.save()
    results["fraud_auc"] = sup["auc_roc"]

    # ── STEP 5: Train Recommendation Engine ───────────────────
    banner(5, "RECOMMENDATION ENGINE")
    from src.models.recommendation_engine import HybridRecommender

    members      = data["members"]
    interactions = members[["member_id", "plan_type"]].copy()
    interactions["rating"] = 1

    recommender = HybridRecommender()
    recommender.fit(members, interactions)
    recommender.save()

    sample   = members.iloc[0]
    recs     = recommender.recommend(
        sample["member_id"],
        {"age": int(sample["age"]), "bmi": float(sample["bmi"]),
         "num_chronic_conditions": int(sample.get("num_chronic_conditions", 0)),
         "tenure_months": int(sample["tenure_months"]),
         "annual_premium": float(sample["annual_premium"]),
         "deductible": int(sample["deductible"])},
    )
    print(f"  Sample recs for {sample['member_id']}: {recs}")

    # ── STEP 6: Evaluate & Check Thresholds ───────────────────
    banner(6, "MODEL EVALUATION")
    MIN_AUC = 0.75
    passed  = True

    print(f"\n  {'Model':<25} {'AUC':<10} {'Status'}")
    print(f"  {'-'*45}")
    for name, auc in results.items():
        status = "PASS" if auc >= MIN_AUC else "FAIL"
        icon   = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name:<23} {auc:<10.4f} {status}")
        if status == "FAIL":
            passed = False

    if not passed:
        print(f"\n  WARNING: Some models below minimum AUC of {MIN_AUC}")
        print("  Consider: more data, feature tuning, hyperparameter search")
    else:
        print(f"\n  All models passed minimum AUC threshold of {MIN_AUC}")

    # ── STEP 7: Save Registry ─────────────────────────────────
    banner(7, "SAVING MODEL REGISTRY")
    Path("models/registry").mkdir(parents=True, exist_ok=True)
    registry = {
        "timestamp":  datetime.now().isoformat(),
        "best_claim": best_model,
        "results":    results,
        "passed":     passed,
    }
    with open("models/registry/run_results.json", "w") as f:
        json.dump(registry, f, indent=2)
    print("  Saved : models/registry/run_results.json")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    mins    = int(elapsed // 60)
    secs    = int(elapsed % 60)

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Time taken : {mins}m {secs}s")
    print(f"  Models     : models/claim_approval/  models/fraud_detection/")
    print(f"  Registry   : models/registry/run_results.json")
    print()
    print("  Next step — Start the API:")
    print("  uvicorn src.api.main:app --reload --port 8000")
    print("  Then open: http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
