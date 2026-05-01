"""
run_recommendation.py
---------------------
Runs the Hybrid Recommendation Engine locally.
Usage: python run_recommendation.py
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from pathlib import Path
from src.models.recommendation_engine import (
    CollaborativeFilter,
    KNNRecommender,
    ContentBasedFilter,
    HybridRecommender,
)

print("=" * 60)
print("  RECOMMENDATION ENGINE TRAINING")
print("=" * 60)

# ── Step 1: Load members data ─────────────────────────────────
print("\nStep 1: Loading members data...")
members_path = Path("data/synthetic/members.csv")

if not members_path.exists():
    print("  ERROR: members.csv not found!")
    print("  Run this first:")
    print("  python src/ingestion/generate_synthetic_data.py")
    sys.exit(1)

members = pd.read_csv(members_path)
print(f"  Loaded  : {len(members):,} members")
print(f"  Plans   : {members['plan_type'].value_counts().to_dict()}")

# ── Step 2: Build interaction matrix ─────────────────────────
print("\nStep 2: Building interaction matrix...")
interactions = pd.DataFrame({
    "member_id": members["member_id"],
    "plan_type": members["plan_type"],
    "rating":    1,
})
print(f"  Interactions : {len(interactions):,} rows")
print(f"  Members      : {interactions['member_id'].nunique():,} unique")
print(f"  Plans        : {interactions['plan_type'].unique().tolist()}")

# ── Step 3: Train Collaborative Filter ───────────────────────
print("\nStep 3: Training Collaborative Filter (SVD)...")
collab = CollaborativeFilter(n_factors=20)
collab.fit(interactions)
print(f"  Fitted on {len(interactions['member_id'].unique()):,} members")
print(f"  Factors : 20")

# Test collaborative filter
sample_id = members["member_id"].iloc[0]
collab_recs = collab.recommend(sample_id, top_n=3)
print(f"  Sample recs for {sample_id}: {collab_recs}")

# ── Step 4: Train KNN Recommender ────────────────────────────
print("\nStep 4: Training KNN Recommender (k=10)...")
knn = KNNRecommender(k=10)
knn.fit(members)
print(f"  Fitted on {len(members):,} members")
print(f"  Features : age, bmi, chronic conditions, tenure, premium, deductible")

# Test KNN
sample_profile = {
    "age":                    45,
    "bmi":                    27.5,
    "num_chronic_conditions": 1,
    "tenure_months":          24,
    "annual_premium":         6000,
    "deductible":             1500,
}
knn_recs = knn.recommend(sample_profile, top_n=3)
print(f"  Sample recs: {knn_recs}")

# ── Step 5: Test Content-Based Filter ────────────────────────
print("\nStep 5: Testing Content-Based Filter...")
content = ContentBasedFilter()

profiles = [
    {"age": 25, "bmi": 22.0, "num_chronic_conditions": 0,
     "income_score": 0.7, "label": "Young healthy low-risk"},
    {"age": 45, "bmi": 28.0, "num_chronic_conditions": 1,
     "income_score": 0.5, "label": "Middle-aged moderate risk"},
    {"age": 65, "bmi": 32.0, "num_chronic_conditions": 3,
     "income_score": 0.3, "label": "Senior high-risk"},
]

for p in profiles:
    label = p.pop("label")
    recs  = content.recommend(p, top_n=3)
    top   = recs[0]["plan"]
    print(f"  {label:<35} → Top plan: {top}")
    p["label"] = label

# ── Step 6: Train Full Hybrid Recommender ────────────────────
print("\nStep 6: Training Full Hybrid Recommender...")
print("  Weights: Collaborative(40%) + KNN(35%) + Content-Based(25%)")
recommender = HybridRecommender()
recommender.fit(members, interactions)
print(f"  Hybrid recommender trained successfully")

# ── Step 7: Sample recommendations ───────────────────────────
print("\nStep 7: Sample Recommendations")
print("-" * 60)

test_cases = [
    {
        "member_id": members["member_id"].iloc[0],
        "profile": {
            "age": 28, "bmi": 23.0,
            "num_chronic_conditions": 0,
            "tenure_months": 6,
            "annual_premium": 3500,
            "deductible": 5000,
            "income_score": 0.8,
        },
        "description": "Young healthy member (28yr, no conditions)",
    },
    {
        "member_id": members["member_id"].iloc[1],
        "profile": {
            "age": 52, "bmi": 29.5,
            "num_chronic_conditions": 2,
            "tenure_months": 48,
            "annual_premium": 9000,
            "deductible": 1500,
            "income_score": 0.4,
        },
        "description": "Middle-aged with conditions (52yr, 2 chronic)",
    },
    {
        "member_id": members["member_id"].iloc[2],
        "profile": {
            "age": 68, "bmi": 31.0,
            "num_chronic_conditions": 4,
            "tenure_months": 96,
            "annual_premium": 15000,
            "deductible": 500,
            "income_score": 0.2,
        },
        "description": "Senior high-risk member (68yr, 4 conditions)",
    },
]

for case in test_cases:
    recs = recommender.recommend(
        case["member_id"],
        case["profile"],
        top_n=3,
    )
    print(f"\n  Member  : {case['description']}")
    print(f"  Results :")
    for i, r in enumerate(recs, 1):
        bar   = "█" * int(r["score"] * 20)
        print(f"    {i}. {r['plan']:<12} score: {r['score']:.3f}  [{bar}]")

# ── Step 8: Save model ────────────────────────────────────────
print("\nStep 8: Saving model...")
Path("models/recommendation").mkdir(parents=True, exist_ok=True)
recommender.save()
print(f"  Saved : models/recommendation/hybrid_recommender.pkl")

# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 60)
print("  RECOMMENDATION ENGINE COMPLETE")
print("=" * 60)
print(f"  Collaborative Filter : SVD on {len(members):,} members")
print(f"  KNN Recommender      : k=10 nearest neighbors")
print(f"  Content-Based        : cosine similarity on risk profile")
print(f"  Hybrid model saved   : models/recommendation/hybrid_recommender.pkl")
print("=" * 60)
