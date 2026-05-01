"""
claim_approval_model.py
-----------------------
Supervised ML models for claim approval prediction & risk scoring.
Models: Logistic Regression, Random Forest, XGBoost, LightGBM
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models/claim_approval")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ClaimApprovalModel:
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = self._build_model()
        self.threshold = 0.5

    def _build_model(self):
        models = {
            "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "rf": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "gbm": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=3,
                use_label_encoder=False,
                eval_metric="auc",
                random_state=42,
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                class_weight="balanced",
                random_state=42,
            ),
        }
        return models[self.model_type]

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        logger.info(
            f"Training {self.model_type} | Train: {len(X_train):,} | Test: {len(X_test):,}"
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        metrics = {
            "model": self.model_type,
            "auc_roc": round(auc, 4),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion": confusion_matrix(y_test, y_pred).tolist(),
        }
        logger.info(f"  AUC-ROC: {auc:.4f}")

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        metrics["cv_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_auc_std"] = round(cv_scores.std(), 4)
        logger.info(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        self.save()
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def risk_score(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(X)
        return pd.DataFrame(
            {
                "approval_probability": proba.round(4),
                "risk_score": (1 - proba).round(4),
                "risk_band": pd.cut(
                    1 - proba,
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=["Very Low", "Low", "Medium", "High", "Very High"],
                ),
                "decision": np.where(proba >= self.threshold, "Approve", "Review"),
            }
        )

    def feature_importance(self, feature_names: list) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self):
        path = MODEL_DIR / f"{self.model_type}_claim_model.pkl"
        joblib.dump(self.model, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, model_type: str = "xgboost") -> "ClaimApprovalModel":
        obj = cls(model_type)
        obj.model = joblib.load(MODEL_DIR / f"{model_type}_claim_model.pkl")
        return obj


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("\n" + "=" * 60)
    print("  CLAIM APPROVAL MODEL TRAINING")
    print("=" * 60)

    # Load data
    from src.ingestion.data_ingestion import LocalIngestion
    from src.preprocessing.feature_engineering import PreprocessingPipeline

    print("\n📥 Loading data...")
    data = LocalIngestion().load_all()

    print("⚙️  Running feature engineering...")
    pipeline = PreprocessingPipeline()
    processed = pipeline.fit_transform_all(data)

    X = processed["claim_features"]
    y = processed["claim_labels"]
    print(f"   Features: {X.shape} | Approved rate: {y.mean():.1%}\n")

    # Train all 4 models
    results = {}
    for model_type in ["logistic", "rf", "xgboost", "lightgbm"]:
        print(f"🤖 Training: {model_type}")
        model = ClaimApprovalModel(model_type=model_type)
        metrics = model.train(X, y)
        results[model_type] = metrics
        print(
            f"   ✓ AUC-ROC: {metrics['auc_roc']} | CV AUC: {metrics['cv_auc_mean']} ± {metrics['cv_auc_std']}\n"
        )

    # Summary
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for name, m in results.items():
        print(
            f"  {name:<20} AUC: {m['auc_roc']}  |  CV: {m['cv_auc_mean']} ± {m['cv_auc_std']}"
        )

    best = max(results, key=lambda k: results[k]["auc_roc"])
    print(f"\n  🏆 Best model: {best} (AUC: {results[best]['auc_roc']})")
    print(f"\n  Models saved to: models/claim_approval/")
    print("=" * 60)
