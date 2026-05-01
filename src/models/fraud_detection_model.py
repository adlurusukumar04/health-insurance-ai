"""
fraud_detection_model.py
------------------------
Unsupervised + supervised fraud detection:
  - Isolation Forest (anomaly detection)
  - K-Means Clustering (behavioral grouping)
  - Association Rule Mining (unusual treatment combos)
  - XGBoost (supervised, when labels available)
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models/fraud_detection")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class FraudDetectionModel:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=19,
            eval_metric="auc",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.fraud_cluster_ids = []

    # ── Isolation Forest ─────────────────────────────────────────────────────
    def train_isolation_forest(self, X: pd.DataFrame) -> dict:
        logger.info(f"Training Isolation Forest on {len(X):,} samples...")
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)

        scores = self.isolation_forest.decision_function(X_scaled)
        preds = self.isolation_forest.predict(X_scaled)  # -1 = anomaly, 1 = normal
        n_anomalies = (preds == -1).sum()
        logger.info(f"  Anomalies detected: {n_anomalies:,} ({n_anomalies/len(X):.1%})")
        return {
            "n_anomalies": int(n_anomalies),
            "anomaly_rate": float(n_anomalies / len(X)),
        }

    def predict_isolation_forest(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler.transform(X)
        preds = self.isolation_forest.predict(X_scaled)
        scores = self.isolation_forest.decision_function(X_scaled)
        # Normalize anomaly score to [0, 1] — higher = more anomalous
        anomaly_score = 1 - (scores - scores.min()) / (
            scores.max() - scores.min() + 1e-9
        )
        return pd.DataFrame(
            {
                "is_anomaly": (preds == -1).astype(int),
                "anomaly_score": anomaly_score.round(4),
                "fraud_flag": (anomaly_score > 0.75).astype(int),
            },
            index=X.index,
        )

    # ── K-Means Clustering ───────────────────────────────────────────────────
    def train_kmeans(self, X: pd.DataFrame, y_fraud: np.ndarray = None) -> dict:
        logger.info(f"Training K-Means clustering on {len(X):,} samples...")
        X_scaled = self.scaler.transform(X)
        self.kmeans.fit(X_scaled)
        cluster_labels = self.kmeans.labels_

        result = {"clusters": {}}
        for c in range(self.kmeans.n_clusters):
            mask = cluster_labels == c
            fraud_rate = float(y_fraud[mask].mean()) if y_fraud is not None else None
            result["clusters"][str(c)] = {
                "size": int(mask.sum()),
                "fraud_rate": round(fraud_rate, 4) if fraud_rate else None,
            }
            if fraud_rate and fraud_rate > 0.15:
                self.fraud_cluster_ids.append(c)

        logger.info(f"  High-fraud clusters: {self.fraud_cluster_ids}")
        return result

    def predict_kmeans(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)
        distances = np.min(self.kmeans.transform(X_scaled), axis=1)
        return pd.DataFrame(
            {
                "cluster_id": clusters,
                "cluster_distance": distances.round(4),
                "high_fraud_cluster": np.isin(clusters, self.fraud_cluster_ids).astype(
                    int
                ),
            },
            index=X.index,
        )

    # ── Supervised XGBoost (when labels available) ───────────────────────────
    def train_supervised(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        from sklearn.model_selection import train_test_split

        logger.info(f"Training supervised XGBoost | fraud rate: {y.mean():.2%}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"  Supervised AUC-ROC: {auc:.4f}")
        return {
            "auc_roc": round(auc, 4),
            "report": classification_report(
                y_test, (y_proba > 0.5).astype(int), output_dict=True
            ),
        }

    # ── Association Rules (unusual treatment combos) ──────────────────────────
    def association_rule_mining(
        self,
        claims_df: pd.DataFrame,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
    ) -> pd.DataFrame:
        """
        Discovers unusual procedure + diagnosis combinations using Apriori.
        Requires mlxtend: pip install mlxtend
        """
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder

            transactions = (
                claims_df.groupby("claim_id")[["diagnosis_code", "procedure_code"]]
                .apply(lambda x: list(x["diagnosis_code"]) + list(x["procedure_code"]))
                .tolist()
            )

            te = TransactionEncoder()
            te_array = te.fit_transform(transactions)
            basket_df = pd.DataFrame(te_array, columns=te.columns_)

            frequent_items = apriori(
                basket_df, min_support=min_support, use_colnames=True
            )
            rules = association_rules(
                frequent_items, metric="confidence", min_threshold=min_confidence
            )
            logger.info(f"  Found {len(rules)} association rules")
            return rules.sort_values("lift", ascending=False)
        except ImportError:
            logger.warning("mlxtend not installed. Skipping association rules.")
            return pd.DataFrame()

    # ── Combined Prediction ──────────────────────────────────────────────────
    def predict_combined(self, X: pd.DataFrame) -> pd.DataFrame:
        iso = self.predict_isolation_forest(X)
        km = self.predict_kmeans(X)
        combo = pd.concat([iso, km], axis=1)
        combo["combined_fraud_score"] = (
            iso["anomaly_score"] * 0.5
            + km["high_fraud_cluster"].astype(float) * 0.3
            + iso["is_anomaly"].astype(float) * 0.2
        ).round(4)
        combo["final_fraud_flag"] = (combo["combined_fraud_score"] > 0.5).astype(int)
        return combo

    def save(self):
        joblib.dump(self, MODEL_DIR / "fraud_detection_model.pkl")
        logger.info("Fraud model saved → {MODEL_DIR}")

    @classmethod
    def load(cls) -> "FraudDetectionModel":
        return joblib.load(MODEL_DIR / "fraud_detection_model.pkl")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("\n" + "=" * 60)
    print("  FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    # Load data
    from src.ingestion.data_ingestion import LocalIngestion
    from src.preprocessing.feature_engineering import PreprocessingPipeline

    print("\n📥 Loading data...")
    data = LocalIngestion().load_all()

    print("⚙️  Running feature engineering...")
    pipeline = PreprocessingPipeline()
    processed = pipeline.fit_transform_all(data)

    X = processed["fraud_features"]
    y = processed["fraud_labels"]
    print(f"   Features: {X.shape} | Fraud rate: {y.mean():.1%}\n")

    model = FraudDetectionModel()

    # Stage 1 — Isolation Forest
    print("🔍 Stage 1: Training Isolation Forest...")
    iso_result = model.train_isolation_forest(X)
    print(
        f"   ✓ Anomaly rate: {iso_result['anomaly_rate']:.1%} | Anomalies found: {iso_result['n_anomalies']:,}\n"
    )

    # Stage 2 — K-Means Clustering
    print("🔵 Stage 2: Training K-Means Clustering...")
    km_result = model.train_kmeans(X, y)
    print(f"   ✓ Clusters created: {len(km_result['clusters'])}")
    for cid, info in km_result["clusters"].items():
        flag = (
            " ← HIGH FRAUD" if info["fraud_rate"] and info["fraud_rate"] > 0.15 else ""
        )
        print(
            f"      Cluster {cid}: {info['size']:,} claims | fraud rate: {info['fraud_rate']:.1%}{flag}"
        )
    print(f"   High-fraud clusters: {model.fraud_cluster_ids}\n")

    # Stage 3 — Supervised XGBoost
    print("🤖 Stage 3: Training Supervised XGBoost...")
    sup_result = model.train_supervised(X, y)
    print(f"   ✓ AUC-ROC: {sup_result['auc_roc']}\n")

    # Test combined prediction
    print("🔗 Testing combined ensemble prediction...")
    sample = X.head(5)
    preds = model.predict_combined(sample)
    print(
        f"   Sample predictions:\n{preds[['anomaly_score','combined_fraud_score','final_fraud_flag']].to_string()}\n"
    )

    # Save model
    model.save()

    print("=" * 60)
    print("  FRAUD DETECTION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Isolation Forest  →  Anomaly rate:    {iso_result['anomaly_rate']:.1%}")
    print(f"  K-Means           →  High-risk clusters: {model.fraud_cluster_ids}")
    print(f"  XGBoost           →  AUC-ROC:         {sup_result['auc_roc']}")
    print(f"\n  Model saved to: models/fraud_detection/")
    print("=" * 60)
