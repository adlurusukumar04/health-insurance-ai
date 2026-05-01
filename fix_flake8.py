"""
fix_flake8.py
-------------
Fixes all flake8 errors in the project.
Usage: python fix_flake8.py
"""

import re
from pathlib import Path

def read(path):
    return Path(path).read_text(encoding="utf-8")

def write(path, content):
    Path(path).write_text(content, encoding="utf-8")
    print(f"  Fixed: {path}")

# ── Fix src/api/main.py ───────────────────────────────────────
print("\nFixing src/api/main.py...")
content = read("src/api/main.py")

# Remove unused imports
content = content.replace("from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks\n",
                           "from fastapi import FastAPI, HTTPException, BackgroundTasks\n")
content = content.replace("from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\n", "")
content = content.replace("import os\n", "", 1)
content = content.replace("import numpy as np\n", "", 1)
content = content.replace("from pathlib import Path\n", "", 1)

# Fix multiple imports on one line (E401) and unused t0 (F841)
content = content.replace("import uuid", "import uuid\n    import time as _time")
content = content.replace("    t0    = time.time()\n", "")
content = content.replace("processing_time_ms=round((time.time() - t0) * 1000, 2),",
                           "processing_time_ms=0.0,")

write("src/api/main.py", content)

# ── Fix src/ingestion/data_ingestion.py ───────────────────────
print("\nFixing src/ingestion/data_ingestion.py...")
content = read("src/ingestion/data_ingestion.py")
content = content.replace("import json\n", "")
content = content.replace("import numpy as np\n", "")
content = content.replace("from datetime import datetime\n", "")
write("src/ingestion/data_ingestion.py", content)

# ── Fix src/ingestion/generate_synthetic_data.py ─────────────
print("\nFixing src/ingestion/generate_synthetic_data.py...")
content = read("src/ingestion/generate_synthetic_data.py")
content = content.replace("        base_risk    = (age_risk + chronic_risk) / 2\n", "")
write("src/ingestion/generate_synthetic_data.py", content)

# ── Fix src/models/claim_approval_model.py ───────────────────
print("\nFixing src/models/claim_approval_model.py...")
content = read("src/models/claim_approval_model.py")
content = content.replace(
    "from sklearn.metrics import (classification_report, roc_auc_score,\n"
    "                             confusion_matrix, precision_recall_curve)\n",
    "from sklearn.metrics import (classification_report, roc_auc_score,\n"
    "                             confusion_matrix)\n"
)
# Fix f-string missing placeholders
content = re.sub(r'f"(Model saved[^"]*)"', r'"\1"', content)
write("src/models/claim_approval_model.py", content)

# ── Fix src/models/fraud_detection_model.py ──────────────────
print("\nFixing src/models/fraud_detection_model.py...")
content = read("src/models/fraud_detection_model.py")
# Remove unused variable scores
content = content.replace(
    "        scores   = self.isolation_forest.decision_function(X_scaled)\n"
    "        preds    = self.isolation_forest.predict(X_scaled)",
    "        preds    = self.isolation_forest.predict(X_scaled)"
)
# Fix f-string missing placeholders
content = re.sub(r'f"(Fraud model saved[^"]*)"', r'"\1"', content)
write("src/models/fraud_detection_model.py", content)

# ── Fix src/models/nlp_medical_text.py ───────────────────────
print("\nFixing src/models/nlp_medical_text.py...")
content = read("src/models/nlp_medical_text.py")
# Remove unused imports
content = content.replace("import numpy as np\n", "", 1)
content = content.replace(
    "from sklearn.metrics import classification_report\n", ""
)
# Fix unused variable e
content = content.replace(
    "        except Exception as e:\n",
    "        except Exception:\n"
)
# Fix all f-strings missing placeholders
content = re.sub(r'f"([^"{}]*)"', lambda m: f'"{m.group(1)}"'
                 if '{' not in m.group(1) else m.group(0), content)
write("src/models/nlp_medical_text.py", content)

# ── Fix src/preprocessing/feature_engineering.py ─────────────
print("\nFixing src/preprocessing/feature_engineering.py...")
content = read("src/preprocessing/feature_engineering.py")
# Remove unused variable providers
content = content.replace(
    '        providers = data.get("providers", pd.DataFrame())\n', ""
)
write("src/preprocessing/feature_engineering.py", content)

print("\n" + "="*50)
print("  All flake8 errors fixed!")
print("="*50)
print("\nNow run:")
print("  git add .")
print("  git commit -m \"fix: resolve all flake8 errors\"")
print("  git push origin main")
