"""
run_api.py
----------
Starts the FastAPI server and tests all 6 endpoints.
Usage: python run_api.py
"""

import sys
import time
import json
import subprocess
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_response(endpoint, response):
    print(f"\n  Endpoint : {endpoint}")
    print(f"  Status   : {response.status_code}")
    try:
        data = response.json()
        print(f"  Response : {json.dumps(data, indent=4)}")
    except Exception:
        print(f"  Response : {response.text}")

def test_all_endpoints():
    print_header("TESTING ALL API ENDPOINTS")

    # ── 1. Health Check ───────────────────────────────────────
    print("\n[1/6] GET /api/v1/health")
    try:
        r = requests.get(f"{BASE_URL}/api/v1/health")
        print_response("GET /api/v1/health", r)
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 2. Models Status ──────────────────────────────────────
    print("\n[2/6] GET /api/v1/models/status")
    try:
        r = requests.get(f"{BASE_URL}/api/v1/models/status")
        print_response("GET /api/v1/models/status", r)
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 3. Claim Prediction ───────────────────────────────────
    print("\n[3/6] POST /api/v1/claims/predict")
    claim_payload = {
        "member_id":        "MBR000001",
        "claim_amount":     5000.00,
        "claim_type":       "Outpatient",
        "diagnosis_code":   "I10",
        "procedure_code":   "99213",
        "num_procedures":   2,
        "days_in_hospital": 0,
        "prior_auth":       1,
        "member_age":       45,
        "member_bmi":       27.5,
        "member_smoker":    0,
        "member_plan":      "Silver",
        "num_chronic":      1,
        "state":            "CA",
    }
    try:
        r = requests.post(f"{BASE_URL}/api/v1/claims/predict", json=claim_payload)
        print_response("POST /api/v1/claims/predict", r)
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 4. Fraud Detection ────────────────────────────────────
    print("\n[4/6] POST /api/v1/fraud/detect")
    fraud_payload = {
        "claim_id":               "CLM0000001",
        "member_id":              "MBR000001",
        "provider_id":            "PRV0001",
        "claim_amount":           50000.00,
        "num_procedures":         15,
        "days_in_hospital":       0,
        "provider_avg_claim":     5000.0,
        "provider_claim_count":   100,
        "provider_unique_members":80,
        "member_claim_count":     5,
        "member_avg_claim":       3000.0,
        "member_total_spend":     15000.0,
    }
    try:
        r = requests.post(f"{BASE_URL}/api/v1/fraud/detect", json=fraud_payload)
        print_response("POST /api/v1/fraud/detect", r)
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 5. NLP Analysis ───────────────────────────────────────
    print("\n[5/6] POST /api/v1/nlp/analyze")
    nlp_payload = {
        "note_id":       "NOTE000001",
        "clinical_note": "Patient presents with hypertension. BP 150/95. Prescribed Lisinopril 10mg. Follow up in 2 weeks.",
    }
    try:
        r = requests.post(f"{BASE_URL}/api/v1/nlp/analyze", json=nlp_payload)
        print_response("POST /api/v1/nlp/analyze", r)
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 6. Plan Recommendation ────────────────────────────────
    print("\n[6/6] POST /api/v1/recommend")
    recommend_payload = {
        "member_id":              "MBR000001",
        "age":                    55,
        "bmi":                    28.0,
        "num_chronic_conditions": 2,
        "tenure_months":          36,
        "annual_premium":         8000.0,
        "deductible":             2000,
        "income_score":           0.5,
    }
    try:
        r = requests.post(f"{BASE_URL}/api/v1/recommend", json=recommend_payload)
        print_response("POST /api/v1/recommend", r)
    except Exception as e:
        print(f"  ERROR: {e}")

    print_header("ALL ENDPOINTS TESTED")
    print("\n  Swagger UI : http://localhost:8000/docs")
    print("  ReDoc      : http://localhost:8000/redoc")
    print()


if __name__ == "__main__":
    print_header("FASTAPI — HEALTH INSURANCE AI PLATFORM")

    # ── Install requests if missing ───────────────────────────
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    # ── Check if API is already running ──────────────────────
    try:
        r = requests.get(f"{BASE_URL}/api/v1/health", timeout=2)
        if r.status_code == 200:
            print("\n  API is already running!")
            test_all_endpoints()
            sys.exit(0)
    except Exception:
        pass

    # ── Start the API in background ───────────────────────────
    print("\n  Starting FastAPI server...")
    print("  URL     : http://localhost:8000")
    print("  Swagger : http://localhost:8000/docs")
    print("\n  Press CTRL+C to stop the server\n")
    print("  Waiting for server to start...")

    # Start uvicorn
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "src.api.main:app",
         "--host", "0.0.0.0",
         "--port", "8000",
         "--reload"],
        env={**__import__("os").environ, "PYTHONPATH": "."}
    )

    # Wait for server to be ready
    for i in range(30):
        time.sleep(1)
        try:
            r = requests.get(f"{BASE_URL}/api/v1/health", timeout=2)
            if r.status_code == 200:
                print(f"  Server ready after {i+1} seconds!")
                break
        except Exception:
            print(f"  Waiting... ({i+1}s)")
    else:
        print("  Server took too long to start.")
        print("  Try running manually:")
        print("  uvicorn src.api.main:app --reload --port 8000")
        server.terminate()
        sys.exit(1)

    # Run tests
    test_all_endpoints()

    # Keep server running
    print("  Server is running. Press CTRL+C to stop.")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\n  Stopping server...")
        server.terminate()
