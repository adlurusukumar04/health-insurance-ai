"""
feature_engineering.py
-----------------------
Feature engineering & preprocessing for all modules:
  - Claim Approval & Risk Scoring
  - Fraud Detection
  - NLP / Medical Text
  - Recommendation Engine
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path("models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Base Transformer ────────────────────────────────────────────────────────
class BaseTransformer:
    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, name: str) -> None:
        joblib.dump(self, ARTIFACTS_DIR / f"{name}.pkl")
        logger.info(f"Saved transformer: {name}.pkl")

    @classmethod
    def load(cls, name: str) -> "BaseTransformer":
        return joblib.load(ARTIFACTS_DIR / f"{name}.pkl")


# ─── Claim Feature Engineering ───────────────────────────────────────────────
class ClaimFeatureEngineer(BaseTransformer):
    """
    Builds features for Claim Approval & Risk Scoring model.
    Input: merged claims + members + providers DataFrame
    Output: feature matrix ready for ML
    """

    CATEGORICAL_COLS = [
        "claim_type",
        "member_plan",
        "state",
        "procedure_code",
        "diagnosis_code",
    ]
    NUMERICAL_COLS = [
        "claim_amount",
        "member_age",
        "member_bmi",
        "num_chronic",
        "num_procedures",
        "days_in_hospital",
        "prior_auth",
        "member_smoker",
    ]
    TARGET_COL = "claim_approved"  # 1 = Approved, 0 = Denied/Pending

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names = []

    def fit(self, df: pd.DataFrame) -> "ClaimFeatureEngineer":
        df = df.copy()
        df = self._base_features(df)

        # Fit imputer
        self.imputer.fit(df[self.NUMERICAL_COLS])

        # Fit label encoders
        for col in self.CATEGORICAL_COLS:
            le = LabelEncoder()
            le.fit(df[col].astype(str).fillna("Unknown"))
            self.label_encoders[col] = le

        # Fit scaler
        num_imputed = self.imputer.transform(df[self.NUMERICAL_COLS])
        cat_encoded = np.column_stack(
            [
                self.label_encoders[c].transform(df[c].astype(str).fillna("Unknown"))
                for c in self.CATEGORICAL_COLS
            ]
        )
        X = np.hstack([num_imputed, cat_encoded])
        self.scaler.fit(X)
        self.feature_names = self.NUMERICAL_COLS + self.CATEGORICAL_COLS
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._base_features(df)

        num_imputed = self.imputer.transform(df[self.NUMERICAL_COLS])
        cat_encoded = np.column_stack(
            [
                self.label_encoders[c].transform(
                    df[c]
                    .astype(str)
                    .fillna("Unknown")
                    .map(
                        lambda x: (
                            x if x in self.label_encoders[c].classes_ else "Unknown"
                        )
                    )
                )
                for c in self.CATEGORICAL_COLS
            ]
        )
        X = np.hstack([num_imputed, cat_encoded])
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=df.index)

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer domain-specific features."""
        df["claim_per_procedure"] = df["claim_amount"] / (df["num_procedures"].clip(1))
        df["risk_score_age"] = (df["member_age"] / 85).clip(0, 1)
        df["risk_score_bmi"] = ((df["member_bmi"] - 18.5) / 40).clip(0, 1)
        df["high_cost_flag"] = (
            df["claim_amount"] > df["claim_amount"].quantile(0.90)
        ).astype(int)
        df["inpatient_flag"] = (df["claim_type"] == "Inpatient").astype(int)
        df["emergency_flag"] = (df["claim_type"] == "Emergency").astype(int)
        df["composite_risk"] = (
            df["risk_score_age"] * 0.3
            + df["risk_score_bmi"] * 0.2
            + (df["num_chronic"] / 3).clip(0, 1) * 0.3
            + df["member_smoker"] * 0.1
            + df["high_cost_flag"] * 0.1
        )

        # Add engineered cols to numerical
        extra = [
            "claim_per_procedure",
            "risk_score_age",
            "risk_score_bmi",
            "high_cost_flag",
            "inpatient_flag",
            "emergency_flag",
            "composite_risk",
        ]
        for col in extra:
            if col not in self.NUMERICAL_COLS:
                self.NUMERICAL_COLS.append(col)

        # Target encoding
        if "claim_status" in df.columns:
            df[self.TARGET_COL] = (df["claim_status"] == "Approved").astype(int)
        return df

    def get_feature_names(self) -> list:
        return self.feature_names


# ─── Fraud Feature Engineering ───────────────────────────────────────────────
class FraudFeatureEngineer(BaseTransformer):
    """
    Builds features for anomaly / fraud detection.
    Designed for unsupervised models (Isolation Forest, K-Means).
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names = []

    def fit(self, df: pd.DataFrame) -> "FraudFeatureEngineer":
        features = self._engineer(df)
        self.feature_names = features.columns.tolist()
        self.scaler.fit(features)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._engineer(df)
        scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, columns=self.feature_names, index=df.index)

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Provider-level aggregations (join required upstream)
        provider_stats = (
            df.groupby("provider_id")
            .agg(
                provider_avg_claim=("claim_amount", "mean"),
                provider_claim_count=("claim_id", "count"),
                provider_unique_members=("member_id", "nunique"),
            )
            .reset_index()
        )
        df = df.merge(provider_stats, on="provider_id", how="left")

        # Member-level velocity
        member_stats = (
            df.groupby("member_id")
            .agg(
                member_claim_count=("claim_id", "count"),
                member_avg_claim=("claim_amount", "mean"),
                member_total_spend=("claim_amount", "sum"),
            )
            .reset_index()
        )
        df = df.merge(member_stats, on="member_id", how="left")

        features = df[
            [
                "claim_amount",
                "num_procedures",
                "days_in_hospital",
                "provider_avg_claim",
                "provider_claim_count",
                "provider_unique_members",
                "member_claim_count",
                "member_avg_claim",
                "member_total_spend",
            ]
        ].fillna(0)

        # Ratio features — strong fraud signals
        features["amount_vs_provider_avg"] = (
            df["claim_amount"] / (df["provider_avg_claim"].clip(1))
        ).fillna(1)
        features["amount_vs_member_avg"] = (
            df["claim_amount"] / (df["member_avg_claim"].clip(1))
        ).fillna(1)
        features["procedure_density"] = (
            df["num_procedures"] / (df["days_in_hospital"].clip(1))
        ).fillna(0)

        return features


# ─── Preprocessing Pipeline ───────────────────────────────────────────────────
class PreprocessingPipeline:
    """
    Orchestrates all preprocessing steps.
    Runs fit/transform for all module engineers.
    """

    def __init__(self):
        self.claim_engineer = ClaimFeatureEngineer()
        self.fraud_engineer = FraudFeatureEngineer()

    def fit_transform_all(self, data: dict) -> dict:
        """
        data: dict with keys 'members', 'claims', 'providers', 'notes'
        Returns dict of feature matrices.
        """
        claims = data["claims"]
        members = data.get("members", pd.DataFrame())

        # Rename columns if they exist under different names (from generate_synthetic_data)
        rename_map = {
            "age": "member_age",
            "bmi": "member_bmi",
            "smoker": "member_smoker",
            "plan_type": "member_plan",
            "num_chronic_conditions": "num_chronic",
        }
        for old_col, new_col in rename_map.items():
            if old_col in claims.columns and new_col not in claims.columns:
                claims = claims.rename(columns={old_col: new_col})

        # Merge claims with member info only if key columns are still missing
        missing = [
            c
            for c in [
                "member_age",
                "member_bmi",
                "member_smoker",
                "member_plan",
                "num_chronic",
            ]
            if c not in claims.columns
        ]
        if missing and not members.empty:
            member_cols = members[
                [
                    "member_id",
                    "age",
                    "bmi",
                    "smoker",
                    "plan_type",
                    "num_chronic_conditions",
                ]
            ].rename(
                columns={
                    "age": "member_age",
                    "bmi": "member_bmi",
                    "smoker": "member_smoker",
                    "plan_type": "member_plan",
                    "num_chronic_conditions": "num_chronic",
                }
            )
            claims = claims.merge(member_cols, on="member_id", how="left")

        # Fill any remaining missing columns with safe defaults
        defaults = {
            "member_age": 40,
            "member_bmi": 25.0,
            "member_smoker": 0,
            "member_plan": "Silver",
            "num_chronic": 0,
        }
        for col, default in defaults.items():
            if col not in claims.columns:
                claims[col] = default

        logger.info("Fitting claim feature engineer...")
        claim_features = self.claim_engineer.fit_transform(claims)
        claim_labels = (claims["claim_status"] == "Approved").astype(int)

        logger.info("Fitting fraud feature engineer...")
        fraud_features = self.fraud_engineer.fit_transform(claims)
        fraud_labels = claims.get("is_fraud", pd.Series(np.zeros(len(claims))))

        # Save artifacts
        self.claim_engineer.save("claim_feature_engineer")
        self.fraud_engineer.save("fraud_feature_engineer")

        logger.info(f"  Claim features: {claim_features.shape}")
        logger.info(f"  Fraud features: {fraud_features.shape}")

        return {
            "claim_features": claim_features,
            "claim_labels": claim_labels,
            "fraud_features": fraud_features,
            "fraud_labels": fraud_labels.values,
            "raw_claims": claims,
        }


if __name__ == "__main__":
    from src.ingestion.data_ingestion import LocalIngestion

    data = LocalIngestion().load_all()
    pipeline = PreprocessingPipeline()
    processed = pipeline.fit_transform_all(data)
    for k, v in processed.items():
        if hasattr(v, "shape"):
            print(f"{k}: {v.shape}")
