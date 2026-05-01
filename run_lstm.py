"""
run_lstm.py
-----------
Runs LSTM diagnosis classifier locally — no GPU needed.
Usage: python run_lstm.py
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from src.models.nlp_medical_text import MedicalTextPreprocessor, LSTMTrainer

print("=" * 50)
print("  NLP — LSTM Diagnosis Classifier")
print("=" * 50)

# ── Step 1: Load clinical notes ───────────────────
print("\nStep 1: Loading clinical notes...")
notes_path = Path("data/synthetic/clinical_notes.csv")

if not notes_path.exists():
    print("  ERROR: clinical_notes.csv not found!")
    print("  Run this first:")
    print("  python src/ingestion/generate_synthetic_data.py")
    sys.exit(1)

notes_df = pd.read_csv(notes_path)
print(f"  Loaded : {len(notes_df):,} notes")
print(f"  Sample : {notes_df['clinical_note'].iloc[0][:80]}")

# ── Step 2: Clean text ────────────────────────────
print("\nStep 2: Cleaning medical text...")
proc = MedicalTextPreprocessor()
notes_df["clean_text"] = proc.batch_clean(notes_df["clinical_note"])
print(f"  Done   : {len(notes_df):,} notes cleaned")
print(f"  Before : {notes_df['clinical_note'].iloc[0][:60]}")
print(f"  After  : {notes_df['clean_text'].iloc[0][:60]}")

# Save cleaned notes
Path("data/processed").mkdir(exist_ok=True)
notes_df.to_csv("data/processed/cleaned_notes.csv", index=False)
print(f"  Saved  : data/processed/cleaned_notes.csv")

# ── Step 3: Train LSTM ────────────────────────────
print("\nStep 3: Training LSTM model...")
texts       = notes_df["clean_text"].tolist()
labels      = notes_df["diagnosis_code"].tolist()
num_classes = len(set(labels))

print(f"  Notes      : {len(texts):,}")
print(f"  Classes    : {num_classes} unique diagnosis codes")
print(f"  Training...\n")

trainer = LSTMTrainer(
    num_classes=num_classes,
    max_vocab=5000,
    max_len=32,
)
history = trainer.train(
    texts=texts,
    labels=labels,
    epochs=5,
    batch_size=32,
)

# ── Step 4: Show results ──────────────────────────
print("\nStep 4: Training Results")
print("-" * 40)
for h in history["history"]:
    filled  = int(h["test_acc"] * 20)
    bar     = "█" * filled + "░" * (20 - filled)
    print(f"  Epoch {h['epoch']}: [{bar}] {h['test_acc']:.1%}")

# ── Step 5: Sentiment analysis (keyword fallback) ─
print("\nStep 5: Sentiment Analysis")
print("-" * 40)
feedback = [
    "The claim process was very smooth and fast. Excellent service!",
    "I have been waiting 3 weeks and still no response. Very frustrating.",
    "The doctor was helpful but the billing was confusing.",
    "Great experience overall. The team was very professional.",
    "My claim was rejected without any explanation. Very disappointed.",
]

pos_words = ["smooth", "fast", "excellent", "great", "professional",
             "helpful", "good", "happy", "satisfied"]
neg_words = ["waiting", "frustrated", "rejected", "disappointed",
             "confusing", "slow", "bad", "terrible", "awful"]

print()
for text in feedback:
    words = text.lower().split()
    pos   = sum(1 for w in pos_words if w in words)
    neg   = sum(1 for w in neg_words if w in words)
    if pos > neg:
        sentiment = "POSITIVE"
        emoji     = "😊"
    elif neg > pos:
        sentiment = "NEGATIVE"
        emoji     = "😞"
    else:
        sentiment = "NEUTRAL"
        emoji     = "😐"
    print(f"  {emoji}  [{sentiment:<8}]  {text[:55]}...")

# ── Summary ───────────────────────────────────────
print()
print("=" * 50)
print("  NLP PIPELINE COMPLETE")
print("=" * 50)
print(f"  Text preprocessing : {len(notes_df):,} notes cleaned")
print(f"  LSTM classifier    : {num_classes} diagnosis classes")
print(f"  Final accuracy     : {history['history'][-1]['test_acc']:.1%}")
print(f"  Model saved        : models/nlp/lstm_model.pt")
print(f"  Cleaned notes      : data/processed/cleaned_notes.csv")
print("=" * 50)
