"""
generate_synthetic_data.py
--------------------------
Generates realistic synthetic health insurance datasets for:
  - Claims data
  - Patient/member profiles
  - Provider records
  - Transaction logs (for fraud detection)
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = Path("data/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
DIAGNOSIS_CODES = [
    "I10",
    "E11",
    "J45",
    "M54",
    "K21",
    "F32",
    "I25",
    "N18",
    "C34",
    "G43",
    "J06",
    "E78",
    "I50",
    "M17",
    "F41",
]
DIAGNOSIS_NAMES = [
    "Hypertension",
    "Type 2 Diabetes",
    "Asthma",
    "Back Pain",
    "GERD",
    "Depression",
    "Coronary Artery Disease",
    "CKD",
    "Lung Cancer",
    "Migraine",
    "Upper Respiratory Infection",
    "Hyperlipidemia",
    "Heart Failure",
    "Osteoarthritis",
    "Anxiety",
]
PROCEDURE_CODES = [
    "99213",
    "93000",
    "71046",
    "80053",
    "85025",
    "36415",
    "99232",
    "99285",
]
STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]
PLAN_TYPES = ["Bronze", "Silver", "Gold", "Platinum"]
PROVIDER_TYPES = ["Hospital", "Clinic", "Specialist", "Lab", "Pharmacy"]
CLAIM_STATUS = ["Approved", "Denied", "Pending"]


# ─── 1. Generate Members ──────────────────────────────────────────────────────
def generate_members(n: int = 10000) -> pd.DataFrame:
    print(f"  Generating {n} member records...")
    ages = np.random.normal(45, 15, n).clip(18, 85).astype(int)
    bmi = np.random.normal(27, 5, n).clip(15, 55).round(1)

    records = []
    for i in range(n):
        chronic = random.randint(0, min(3, ages[i] // 25))
        diag_indices = random.sample(
            range(len(DIAGNOSIS_CODES)), k=min(chronic, len(DIAGNOSIS_CODES))
        )
        records.append(
            {
                "member_id": f"MBR{i+1:06d}",
                "age": ages[i],
                "gender": random.choice(["M", "F", "Other"]),
                "state": random.choice(STATES),
                "bmi": bmi[i],
                "smoker": random.choices([0, 1], weights=[0.82, 0.18])[0],
                "plan_type": random.choice(PLAN_TYPES),
                "tenure_months": random.randint(1, 120),
                "num_chronic_conditions": chronic,
                "primary_diagnosis": (
                    DIAGNOSIS_CODES[diag_indices[0]] if diag_indices else "None"
                ),
                "annual_premium": round(random.uniform(3000, 18000), 2),
                "deductible": random.choice([500, 1000, 2000, 3500, 5000]),
                "member_since": (
                    datetime.now() - timedelta(days=random.randint(30, 3650))
                ).strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(records)


# ─── 2. Generate Claims ───────────────────────────────────────────────────────
def generate_claims(members: pd.DataFrame, n: int = 50000) -> pd.DataFrame:
    print(f"  Generating {n} claim records...")
    member_ids = members["member_id"].tolist()

    records = []
    for i in range(n):
        member = members[members["member_id"] == random.choice(member_ids)].iloc[0]
        fraud_flag = 1 if random.random() < 0.05 else 0  # 5% fraud rate
        amount = round(np.random.lognormal(7, 1.2), 2)
        if fraud_flag:
            amount *= random.uniform(3, 10)

        # Claim approval logic
        denial_prob = 0.15 + (0.1 * chronic_risk) + (0.2 * fraud_flag)
        if random.random() < denial_prob:
            status = "Denied"
        elif random.random() < 0.08:
            status = "Pending"
        else:
            status = "Approved"

        claim_date = datetime.now() - timedelta(days=random.randint(1, 730))
        records.append(
            {
                "claim_id": f"CLM{i+1:07d}",
                "member_id": member["member_id"],
                "provider_id": f"PRV{random.randint(1, 500):04d}",
                "claim_date": claim_date.strftime("%Y-%m-%d"),
                "claim_type": random.choice(
                    ["Inpatient", "Outpatient", "Emergency", "Pharmacy", "Lab"]
                ),
                "diagnosis_code": random.choice(DIAGNOSIS_CODES),
                "procedure_code": random.choice(PROCEDURE_CODES),
                "claim_amount": amount,
                "approved_amount": (
                    round(amount * random.uniform(0.6, 1.0), 2)
                    if status == "Approved"
                    else 0
                ),
                "claim_status": status,
                "days_in_hospital": (
                    random.randint(0, 14) if random.random() < 0.3 else 0
                ),
                "num_procedures": random.randint(1, 8),
                "prior_auth": random.choice([0, 1]),
                "is_fraud": fraud_flag,
                "member_age": member["age"],
                "member_bmi": member["bmi"],
                "member_smoker": member["smoker"],
                "member_plan": member["plan_type"],
                "num_chronic": member["num_chronic_conditions"],
                "state": member["state"],
            }
        )
    return pd.DataFrame(records)


# ─── 3. Generate Providers ────────────────────────────────────────────────────
def generate_providers(n: int = 500) -> pd.DataFrame:
    print(f"  Generating {n} provider records...")
    records = []
    for i in range(n):
        records.append(
            {
                "provider_id": f"PRV{i+1:04d}",
                "provider_name": f"Provider_{i+1}",
                "provider_type": random.choice(PROVIDER_TYPES),
                "state": random.choice(STATES),
                "avg_claim_amount": round(random.uniform(500, 15000), 2),
                "total_claims": random.randint(10, 5000),
                "fraud_rate": round(random.uniform(0.0, 0.15), 4),
                "accreditation": random.choice(["JCAHO", "AAAHC", "DNV", "None"]),
                "years_active": random.randint(1, 30),
            }
        )
    return pd.DataFrame(records)


# ─── 4. Generate Clinical Notes (NLP) ────────────────────────────────────────
def generate_clinical_notes(n: int = 5000) -> pd.DataFrame:
    print(f"  Generating {n} clinical notes...")
    templates = [
        "Patient presents with {diag}. BP {bp}. Prescribed {med}. Follow-up in {days} days.",
        "Chief complaint: {symptom}. Diagnosis: {diag}. Labs ordered: CBC, CMP. Treatment: {med}.",
        "Follow-up visit for {diag}. Patient reports {outcome}. Medication adjusted.",
        "{age}yo {gender} with history of {diag} presenting with acute exacerbation.",
        "Post-operative visit. Procedure: {proc}. Wound healing well. No complications noted.",
    ]
    meds = [
        "Metformin 500mg",
        "Lisinopril 10mg",
        "Atorvastatin 20mg",
        "Albuterol inhaler",
        "Omeprazole 20mg",
        "Sertraline 50mg",
    ]
    symptoms = [
        "chest pain",
        "shortness of breath",
        "fatigue",
        "dizziness",
        "nausea",
        "back pain",
        "headache",
        "swelling",
    ]

    records = []
    for i in range(n):
        diag_idx = random.randint(0, len(DIAGNOSIS_CODES) - 1)
        template = random.choice(templates)
        note = template.format(
            diag=DIAGNOSIS_NAMES[diag_idx],
            bp=f"{random.randint(110,160)}/{random.randint(70,100)}",
            med=random.choice(meds),
            days=random.choice([7, 14, 30, 90]),
            symptom=random.choice(symptoms),
            outcome=random.choice(["improvement", "no change", "worsening"]),
            age=random.randint(25, 80),
            gender=random.choice(["male", "female"]),
            proc=random.choice(PROCEDURE_CODES),
        )
        records.append(
            {
                "note_id": f"NOTE{i+1:06d}",
                "member_id": f"MBR{random.randint(1, 10000):06d}",
                "note_date": (
                    datetime.now() - timedelta(days=random.randint(1, 365))
                ).strftime("%Y-%m-%d"),
                "clinical_note": note,
                "diagnosis_code": DIAGNOSIS_CODES[diag_idx],
                "diagnosis_label": DIAGNOSIS_NAMES[diag_idx],
                "sentiment_label": random.choice(["positive", "neutral", "negative"]),
            }
        )
    return pd.DataFrame(records)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    print("\n🔄 Generating synthetic health insurance data...\n")

    members = generate_members(args.members)
    claims = generate_claims(members, args.claims)
    providers = generate_providers(args.providers)
    notes = generate_clinical_notes(args.notes)

    # Save to CSV
    members.to_csv(OUTPUT_DIR / "members.csv", index=False)
    claims.to_csv(OUTPUT_DIR / "claims.csv", index=False)
    providers.to_csv(OUTPUT_DIR / "providers.csv", index=False)
    notes.to_csv(OUTPUT_DIR / "clinical_notes.csv", index=False)

    # Summary
    print("\n✅ Datasets saved to data/synthetic/")
    print(f"   members.csv      → {len(members):,} rows")
    print(
        f"   claims.csv       → {len(claims):,} rows  | fraud rate: {claims['is_fraud'].mean():.1%}"
    )
    print(f"   providers.csv    → {len(providers):,} rows")
    print(f"   clinical_notes.csv → {len(notes):,} rows\n")

    stats = {
        "members": len(members),
        "claims": len(claims),
        "fraud_rate": float(claims["is_fraud"].mean()),
        "approval_rate": float((claims["claim_status"] == "Approved").mean()),
        "providers": len(providers),
        "clinical_notes": len(notes),
    }
    with open(OUTPUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic health insurance data"
    )
    parser.add_argument("--members", type=int, default=10000)
    parser.add_argument("--claims", type=int, default=50000)
    parser.add_argument("--providers", type=int, default=500)
    parser.add_argument("--notes", type=int, default=5000)
    args = parser.parse_args()
    main(args)
