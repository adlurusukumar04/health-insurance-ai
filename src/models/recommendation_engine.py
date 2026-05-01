"""
recommendation_engine.py
-------------------------
Personalized insurance plan recommendation:
  - Collaborative Filtering (Matrix Factorization)
  - K-Nearest Neighbors (user-based)
  - Content-Based Filtering (member profile similarity)
  - Hybrid Ensemble
"""

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models/recommendation")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PLAN_FEATURES = {
    "Bronze": {"premium_score": 1, "coverage_score": 2, "deductible_score": 4},
    "Silver": {"premium_score": 2, "coverage_score": 3, "deductible_score": 3},
    "Gold": {"premium_score": 3, "coverage_score": 4, "deductible_score": 2},
    "Platinum": {"premium_score": 4, "coverage_score": 5, "deductible_score": 1},
}


# ─── Collaborative Filtering (SVD / Matrix Factorization) ─────────────────────
class CollaborativeFilter:
    """
    Implicit matrix factorization using scipy SVD on
    member × plan_type interaction matrix.
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.plan_index = {}
        self.member_index = {}

    def fit(self, interactions: pd.DataFrame) -> "CollaborativeFilter":
        """
        interactions: DataFrame with columns [member_id, plan_type, rating]
        rating = 1 (enrolled) or satisfaction_score (1–5)
        """
        from scipy.sparse.linalg import svds
        from scipy.sparse import csr_matrix

        # Build interaction matrix
        members = interactions["member_id"].unique()
        plans = interactions["plan_type"].unique()
        self.member_index = {m: i for i, m in enumerate(members)}
        self.plan_index = {p: i for i, p in enumerate(plans)}

        rows = interactions["member_id"].map(self.member_index)
        cols = interactions["plan_type"].map(self.plan_index)
        data = interactions["rating"].fillna(1).values

        matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(members), len(plans)), dtype=float
        )

        # SVD decomposition
        k = min(self.n_factors, min(matrix.shape) - 1)
        U, sigma, Vt = svds(matrix, k=k)
        self.user_factors = U * sigma[np.newaxis, :]
        self.item_factors = Vt.T

        logger.info(
            f"CollaborativeFilter fitted | {len(members)} members × {len(plans)} plans | k={k}"
        )
        return self

    def recommend(self, member_id: str, top_n: int = 3) -> list:
        if member_id not in self.member_index:
            return list(self.plan_index.keys())[:top_n]
        idx = self.member_index[member_id]
        scores = self.user_factors[idx] @ self.item_factors.T
        top = np.argsort(scores)[::-1][:top_n]
        plans = {v: k for k, v in self.plan_index.items()}
        return [{"plan": plans[i], "score": float(scores[i])} for i in top]


# ─── KNN User-Based ───────────────────────────────────────────────────────────
class KNNRecommender:
    def __init__(self, k: int = 10):
        self.k = k
        self.knn = NearestNeighbors(
            n_neighbors=k + 1, metric="cosine", algorithm="brute"
        )
        self.scaler = StandardScaler()
        self.member_ids = []
        self.member_plans = {}

    def fit(self, members: pd.DataFrame) -> "KNNRecommender":
        """
        members: DataFrame with member profile features + current plan
        """
        feature_cols = [
            "age",
            "bmi",
            "num_chronic_conditions",
            "tenure_months",
            "annual_premium",
            "deductible",
        ]
        feature_cols = [c for c in feature_cols if c in members.columns]

        self.member_ids = members["member_id"].tolist()
        self.member_plans = dict(
            zip(
                members["member_id"],
                members.get("plan_type", ["Silver"] * len(members)),
            )
        )

        X = members[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.knn.fit(X_scaled)
        self.X_scaled = X_scaled
        logger.info(
            f"KNN fitted on {len(members):,} members with {len(feature_cols)} features"
        )
        return self

    def recommend(self, member_profile: dict, top_n: int = 3) -> list:
        feature_cols = [
            "age",
            "bmi",
            "num_chronic_conditions",
            "tenure_months",
            "annual_premium",
            "deductible",
        ]
        x = np.array([[member_profile.get(f, 0) for f in feature_cols]])
        x_scaled = self.scaler.transform(x)

        distances, indices = self.knn.kneighbors(x_scaled)
        neighbor_ids = [self.member_ids[i] for i in indices[0][1:]]  # skip self

        # Aggregate plan preferences from neighbors
        plan_scores = {}
        for nid, dist in zip(neighbor_ids, distances[0][1:]):
            plan = self.member_plans.get(nid, "Silver")
            weight = 1 / (dist + 1e-9)
            plan_scores[plan] = plan_scores.get(plan, 0) + weight

        sorted_plans = sorted(plan_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"plan": p, "score": round(float(s), 4)} for p, s in sorted_plans[:top_n]
        ]


# ─── Content-Based Filter ─────────────────────────────────────────────────────
class ContentBasedFilter:
    """
    Matches member risk profile to plan features using cosine similarity.
    """

    def recommend(self, member_profile: dict, top_n: int = 3) -> list:
        # Member feature vector
        age_norm = member_profile.get("age", 40) / 85
        chronic_norm = member_profile.get("num_chronic_conditions", 0) / 5
        bmi_norm = (member_profile.get("bmi", 25) - 15) / 40
        income_norm = member_profile.get("income_score", 0.5)  # 0=low, 1=high
        risk_score = age_norm * 0.3 + chronic_norm * 0.4 + bmi_norm * 0.2 + 0.1

        member_vec = np.array([[risk_score, 1 - income_norm, income_norm]])

        plan_vecs, plan_names = [], []
        for plan, feats in PLAN_FEATURES.items():
            # plan_vec: [risk_coverage_need, affordability, premium_willingness]
            plan_vecs.append(
                [
                    feats["coverage_score"] / 5,
                    (5 - feats["deductible_score"]) / 5,
                    feats["premium_score"] / 5,
                ]
            )
            plan_names.append(plan)

        sims = cosine_similarity(member_vec, np.array(plan_vecs))[0]
        sorted_idx = np.argsort(sims)[::-1][:top_n]
        return [
            {"plan": plan_names[i], "score": round(float(sims[i]), 4)}
            for i in sorted_idx
        ]


# ─── Hybrid Recommender ───────────────────────────────────────────────────────
class HybridRecommender:
    """
    Combines Collaborative Filtering + KNN + Content-Based with weighted voting.
    """

    def __init__(self):
        self.collab = CollaborativeFilter()
        self.knn = KNNRecommender()
        self.content = ContentBasedFilter()
        self.weights = {"collab": 0.4, "knn": 0.35, "content": 0.25}

    def fit(
        self, members: pd.DataFrame, interactions: pd.DataFrame
    ) -> "HybridRecommender":
        self.collab.fit(interactions)
        self.knn.fit(members)
        logger.info("HybridRecommender fitted.")
        return self

    def recommend(self, member_id: str, member_profile: dict, top_n: int = 3) -> list:
        collab_recs = self.collab.recommend(member_id, top_n=4)
        knn_recs = self.knn.recommend(member_profile, top_n=4)
        content_recs = self.content.recommend(member_profile, top_n=4)

        # Aggregate with weights
        plan_scores = {}
        for rec_list, weight_key in [
            (collab_recs, "collab"),
            (knn_recs, "knn"),
            (content_recs, "content"),
        ]:
            w = self.weights[weight_key]
            for rec in rec_list:
                plan = rec["plan"]
                plan_scores[plan] = plan_scores.get(plan, 0) + rec["score"] * w

        sorted_plans = sorted(plan_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"plan": p, "score": round(float(s), 4)} for p, s in sorted_plans[:top_n]
        ]

    def save(self):
        joblib.dump(self, MODEL_DIR / "hybrid_recommender.pkl")
        logger.info(f"Recommender saved → {MODEL_DIR}")

    @classmethod
    def load(cls) -> "HybridRecommender":
        return joblib.load(MODEL_DIR / "hybrid_recommender.pkl")
