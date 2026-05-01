"""
run_pipeline.py
---------------
Master orchestrator for the full ML pipeline:
  Step 1 → Data Ingestion
  Step 2 → Preprocessing & Feature Engineering
  Step 3 → Train Claim Approval Model
  Step 4 → Train Fraud Detection Model
  Step 5 → Train Recommendation Engine
  Step 6 → Evaluate & Register Models
  Step 7 → Export artifacts for API

Usage:
  python src/pipelines/run_pipeline.py --module all
  python src/pipelines/run_pipeline.py --module claim
  python src/pipelines/run_pipeline.py --module fraud
  python src/pipelines/run_pipeline.py --module recommend
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)
Path("logs").mkdir(exist_ok=True)
Path("models/registry").mkdir(parents=True, exist_ok=True)


def step_banner(step: int, title: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"  STEP {step}: {title}")
    logger.info(f"{'='*60}")


def run_ingestion() -> dict:
    step_banner(1, "DATA INGESTION")
    from src.ingestion.data_ingestion import DataIngestionFactory

    data = DataIngestionFactory.get_data(source="local")
    for name, df in data.items():
        logger.info(f"  {name}: {df.shape}")
    return data


def run_preprocessing(data: dict) -> dict:
    step_banner(2, "PREPROCESSING & FEATURE ENGINEERING")
    from src.preprocessing.feature_engineering import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    processed = pipeline.fit_transform_all(data)
    logger.info(f"  Claim features:  {processed['claim_features'].shape}")
    logger.info(f"  Fraud features:  {processed['fraud_features'].shape}")
    return processed


def run_claim_model(processed: dict) -> dict:
    step_banner(3, "CLAIM APPROVAL MODEL TRAINING")
    from src.models.claim_approval_model import ClaimApprovalModel

    results = {}
    for model_type in ["logistic", "rf", "xgboost", "lightgbm"]:
        logger.info(f"  Training: {model_type}")
        model = ClaimApprovalModel(model_type=model_type)
        metrics = model.train(processed["claim_features"], processed["claim_labels"])
        results[model_type] = metrics
        logger.info(
            f"    AUC: {metrics['auc_roc']} | CV AUC: {metrics['cv_auc_mean']} ± {metrics['cv_auc_std']}"
        )

    best = max(results, key=lambda k: results[k]["auc_roc"])
    logger.info(f"\n  🏆 Best model: {best} (AUC: {results[best]['auc_roc']})")
    return results


def run_fraud_model(processed: dict) -> dict:
    step_banner(4, "FRAUD DETECTION MODEL TRAINING")
    from src.models.fraud_detection_model import FraudDetectionModel

    model = FraudDetectionModel()
    iso_result = model.train_isolation_forest(processed["fraud_features"])
    logger.info(f"  Isolation Forest → anomaly rate: {iso_result['anomaly_rate']:.2%}")

    km_result = model.train_kmeans(
        processed["fraud_features"], processed["fraud_labels"]
    )
    logger.info(
        f"  K-Means → {len(km_result['clusters'])} clusters | fraud clusters: {model.fraud_cluster_ids}"
    )

    # Supervised XGBoost (uses ground-truth fraud labels)
    sup_result = model.train_supervised(
        processed["fraud_features"], processed["fraud_labels"]
    )
    logger.info(f"  XGBoost supervised → AUC: {sup_result['auc_roc']}")

    model.save()
    return {
        "isolation_forest": iso_result,
        "kmeans": km_result,
        "supervised": sup_result,
    }


def run_recommendation_model(data: dict) -> dict:
    step_banner(5, "RECOMMENDATION ENGINE TRAINING")
    import pandas as pd

    from src.models.recommendation_engine import HybridRecommender

    members = data["members"]
    # Create synthetic interaction matrix from plan enrollment
    interactions = pd.DataFrame(
        {
            "member_id": members["member_id"],
            "plan_type": members["plan_type"],
            "rating": 1,  # Implicit feedback (enrolled = 1)
        }
    )
    recommender = HybridRecommender()
    recommender.fit(members, interactions)
    recommender.save()

    # Smoke test
    sample = members.iloc[0]
    recs = recommender.recommend(
        sample["member_id"],
        {
            "age": sample["age"],
            "bmi": sample["bmi"],
            "num_chronic_conditions": sample.get("num_chronic_conditions", 0),
            "tenure_months": sample["tenure_months"],
            "annual_premium": sample["annual_premium"],
            "deductible": sample["deductible"],
        },
    )
    logger.info(f"  Sample recommendation for {sample['member_id']}: {recs}")
    return {"sample_recommendations": recs}


def register_models(results: dict):
    step_banner(6, "MODEL REGISTRY")
    registry = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    path = Path("models/registry/run_results.json")
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    logger.info(f"  Results saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        default="all",
        choices=["all", "claim", "fraud", "recommend", "ingest"],
    )
    args = parser.parse_args()

    logger.info("\n" + "🏥 " * 20)
    logger.info("  HEALTH INSURANCE AI PLATFORM – LUMEN TECHNOLOGIES")
    logger.info("  Full ML Pipeline Starting...")
    logger.info("🏥 " * 20 + "\n")

    t_start = time.time()
    results = {}

    # Always run ingestion + preprocessing
    data = run_ingestion()
    processed = run_preprocessing(data)

    if args.module in ("all", "claim"):
        results["claim"] = run_claim_model(processed)

    if args.module in ("all", "fraud"):
        results["fraud"] = run_fraud_model(processed)

    if args.module in ("all", "recommend"):
        results["recommend"] = run_recommendation_model(data)

    register_models(results)

    elapsed = time.time() - t_start
    logger.info(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    logger.info("  Run `uvicorn src.api.main:app --reload` to start the API")


if __name__ == "__main__":
    main()
