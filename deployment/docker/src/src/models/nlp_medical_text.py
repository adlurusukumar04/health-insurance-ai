"""
nlp_medical_text.py
--------------------
NLP pipeline for medical text analysis:
  - Text cleaning & tokenization
  - Named Entity Recognition (NER)
  - BERT-based diagnosis classification
  - LSTM sequence model
  - Sentiment analysis from patient feedback
"""

import re
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models/nlp")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Text Preprocessor ───────────────────────────────────────────────────────
class MedicalTextPreprocessor:
    MEDICAL_ABBREVS = {
        "pt": "patient", "hx": "history", "dx": "diagnosis",
        "tx": "treatment", "rx": "prescription", "htn": "hypertension",
        "dm": "diabetes mellitus", "cad": "coronary artery disease",
        "ckd": "chronic kidney disease", "bp": "blood pressure",
        "hr": "heart rate", "sob": "shortness of breath",
        "cp": "chest pain", "n/v": "nausea vomiting",
    }

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        # Expand abbreviations
        for abbrev, full in self.MEDICAL_ABBREVS.items():
            text = re.sub(rf'\b{abbrev}\b', full, text)
        # Remove special chars except medical punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\-\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def batch_clean(self, texts: pd.Series) -> pd.Series:
        return texts.apply(self.clean)


# ─── BERT Diagnosis Classifier ────────────────────────────────────────────────
class BERTDiagnosisClassifier:
    """
    Fine-tunes Bio_ClinicalBERT for multi-class diagnosis classification.
    Uses HuggingFace Transformers.
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 num_labels: int = 15, max_length: int = 128):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = LabelEncoder()
        self.tokenizer  = None
        self.model      = None

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        logger.info(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        ).to(self.device)

    def train(self, texts: list, labels: list, epochs: int = 3, batch_size: int = 16) -> dict:
        from torch.utils.data import Dataset, DataLoader
        from transformers import AdamW, get_linear_schedule_with_warmup

        self._load_model()
        encoded_labels = self.label_encoder.fit_transform(labels)

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.encodings = tokenizer(texts, truncation=True, padding=True,
                                           max_length=max_length, return_tensors="pt")
                self.labels = torch.tensor(labels, dtype=torch.long)
            def __len__(self): return len(self.labels)
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]

        texts_train, texts_val, y_train, y_val = train_test_split(
            texts, encoded_labels, test_size=0.15, stratify=encoded_labels, random_state=42)

        train_ds = TextDataset(texts_train, y_train, self.tokenizer, self.max_length)
        val_ds   = TextDataset(texts_val,   y_val,   self.tokenizer, self.max_length)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dl) // 10,
            num_training_steps=len(train_dl) * epochs,
        )

        history = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_inputs, batch_labels in train_dl:
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(self.device)
                outputs = self.model(**batch_inputs, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dl)
            val_acc  = self._evaluate(val_dl)
            logger.info(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            history.append({"epoch": epoch + 1, "loss": avg_loss, "val_acc": val_acc})

        self.save_bert()
        return {"history": history}

    def _evaluate(self, dataloader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                outputs = self.model(**inputs)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def predict(self, texts: list) -> pd.DataFrame:
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**encodings).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        labels = self.label_encoder.inverse_transform(preds)
        return pd.DataFrame({
            "predicted_diagnosis": labels,
            "confidence":          probs.max(axis=1).round(4),
        })

    def save_bert(self):
        self.model.save_pretrained(MODEL_DIR / "bert_diagnosis")
        self.tokenizer.save_pretrained(MODEL_DIR / "bert_diagnosis")
        import joblib
        joblib.dump(self.label_encoder, MODEL_DIR / "bert_label_encoder.pkl")
        logger.info(f"BERT model saved → {MODEL_DIR}/bert_diagnosis/")


# ─── LSTM Model for Sequence Classification ──────────────────────────────────
class LSTMDiagnosisModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 15, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use last hidden state from both directions
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(out)


class LSTMTrainer:
    def __init__(self, num_classes: int = 15, max_vocab: int = 10000, max_len: int = 64):
        self.num_classes = num_classes
        self.max_vocab   = max_vocab
        self.max_len     = max_len
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab       = {}
        self.label_encoder = LabelEncoder()

    def _build_vocab(self, texts: list):
        from collections import Counter
        all_words = " ".join(texts).split()
        word_freq = Counter(all_words)
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in word_freq.most_common(self.max_vocab - 2):
            self.vocab[word] = len(self.vocab)

    def _encode(self, text: str) -> list:
        tokens = text.split()[:self.max_len]
        ids = [self.vocab.get(w, 1) for w in tokens]
        return ids + [0] * (self.max_len - len(ids))

    def train(self, texts: list, labels: list, epochs: int = 10, batch_size: int = 64) -> dict:
        from torch.utils.data import TensorDataset, DataLoader

        self._build_vocab(texts)
        y = self.label_encoder.fit_transform(labels)
        X = torch.tensor([self._encode(t) for t in texts], dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_tensor, test_size=0.2, random_state=42)

        train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_dl  = DataLoader(TensorDataset(X_test, y_test),  batch_size=batch_size)

        self.model = LSTMDiagnosisModel(
            vocab_size=len(self.vocab), num_classes=self.num_classes
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        history = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_b, y_b in train_dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            acc = self._eval_accuracy(test_dl)
            logger.info(f"  LSTM Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_dl):.4f} | Test Acc: {acc:.4f}")
            history.append({"epoch": epoch+1, "test_acc": acc})

        torch.save(self.model.state_dict(), MODEL_DIR / "lstm_model.pt")
        return {"history": history}

    def _eval_accuracy(self, dl) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in dl:
                X_b = X_b.to(self.device)
                preds = self.model(X_b).argmax(dim=-1).cpu()
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
        return correct / total


# ─── Sentiment Analyser ───────────────────────────────────────────────────────
class SentimentAnalyser:
    """
    Classifies customer feedback as positive / neutral / negative.
    Uses fine-tuned DistilBERT for speed.
    """

    def __init__(self):
        self.pipeline = None

    def load(self):
        from transformers import pipeline
        self.pipeline = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1,
        )

    def predict(self, texts: list) -> pd.DataFrame:
        if self.pipeline is None:
            self.load()
        results = self.pipeline(texts, truncation=True, max_length=512)
        return pd.DataFrame({
            "text":      texts,
            "sentiment": [r["label"] for r in results],
            "score":     [round(r["score"], 4) for r in results],
        })


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n" + "="*60)
    print("  NLP MEDICAL TEXT ANALYSIS")
    print("="*60)

    # ── Step 1: Text Preprocessing ────────────────────────────
    print("\n📝 Step 1: Medical Text Preprocessor")
    print("-"*40)
    preprocessor = MedicalTextPreprocessor()

    test_notes = [
        "Pt presents w/ HTN. BP 145/90. Dx: DM type 2. Tx: Metformin 500mg.",
        "Hx of CAD and CKD. Pt reports sob and cp. HR 95. Rx: Lisinopril.",
        "F/U visit for asthma. Pt using albuterol inhaler. No n/v reported.",
        "Dx: Depression. Pt reports fatigue and low mood. Tx: Sertraline 50mg.",
        "Post-op visit. Procedure: 99213. Wound healing well. No complications.",
    ]

    print("\n  Original → Cleaned:")
    for note in test_notes:
        cleaned = preprocessor.clean(note)
        print(f"\n  ORIGINAL: {note}")
        print(f"  CLEANED:  {cleaned}")

    # ── Step 2: Load clinical notes ───────────────────────────
    print("\n\n📂 Step 2: Loading Clinical Notes Dataset")
    print("-"*40)
    import pandas as pd
    from pathlib import Path

    notes_path = Path("data/synthetic/clinical_notes.csv")
    if notes_path.exists():
        notes_df = pd.read_csv(notes_path)
        print(f"  ✓ Loaded {len(notes_df):,} clinical notes")
        print(f"  Columns: {list(notes_df.columns)}")
        print(f"\n  Sample note:\n  {notes_df['clinical_note'].iloc[0]}")

        # Clean all notes
        print(f"\n  Cleaning all {len(notes_df):,} notes...")
        notes_df["clean_text"] = preprocessor.batch_clean(notes_df["clinical_note"])
        print(f"  ✓ Cleaning complete")
        print(f"\n  Sample cleaned:\n  {notes_df['clean_text'].iloc[0]}")
    else:
        print("  ⚠ clinical_notes.csv not found — using test notes")
        import pandas as pd
        notes_df = pd.DataFrame({
            "clinical_note":   test_notes,
            "diagnosis_code":  ["E11", "I25", "J45", "F32", "J06"],
            "diagnosis_label": ["Diabetes","CAD","Asthma","Depression","URI"],
        })
        notes_df["clean_text"] = preprocessor.batch_clean(notes_df["clinical_note"])

    # ── Step 3: LSTM Training (lightweight — no GPU needed) ───
    print("\n\n🤖 Step 3: LSTM Diagnosis Classifier")
    print("-"*40)
    print("  Training LSTM model (faster than BERT, no GPU needed)...")

    trainer = LSTMTrainer(num_classes=len(notes_df["diagnosis_code"].unique()),
                          max_vocab=5000, max_len=32)
    history = trainer.train(
        texts=notes_df["clean_text"].tolist(),
        labels=notes_df["diagnosis_code"].tolist(),
        epochs=5,
        batch_size=32,
    )
    print(f"\n  Training history:")
    for h in history["history"]:
        print(f"    Epoch {h['epoch']}: Test Accuracy = {h['test_acc']:.4f}")

    # ── Step 4: Sentiment Analysis ────────────────────────────
    print("\n\n😊 Step 4: Sentiment Analysis")
    print("-"*40)
    feedback_samples = [
        "The claim process was very smooth and fast. Excellent service!",
        "I have been waiting 3 weeks and still no response. Very frustrating.",
        "The doctor was helpful but the billing was confusing.",
        "Great experience overall. The team was very professional.",
        "My claim was rejected without any explanation. Very disappointed.",
    ]

    print("  Running sentiment analysis on customer feedback...\n")
    analyser = SentimentAnalyser()
    try:
        analyser.load()
        results = analyser.predict(feedback_samples)
        for _, row in results.iterrows():
            emoji = "😊" if row["sentiment"] == "POSITIVE" else "😞"
            print(f"  {emoji} [{row['sentiment']}] ({row['score']:.2f}) — {row['text'][:60]}...")
    except Exception as e:
        print(f"  ⚠ Sentiment model needs internet to download (~250MB)")
        print(f"  Running keyword-based fallback instead...\n")
        for text in feedback_samples:
            positive_words = ["smooth","fast","excellent","great","professional","helpful"]
            negative_words = ["waiting","frustrating","rejected","disappointed","confusing"]
            pos = sum(1 for w in positive_words if w in text.lower())
            neg = sum(1 for w in negative_words if w in text.lower())
            sentiment = "POSITIVE" if pos > neg else "NEGATIVE" if neg > pos else "NEUTRAL"
            emoji = "😊" if sentiment == "POSITIVE" else "😞" if sentiment == "NEGATIVE" else "😐"
            print(f"  {emoji} [{sentiment}] — {text[:60]}...")

    # ── Step 5: BERT Note (optional) ──────────────────────────
    print("\n\n💡 Step 5: BERT Classifier Note")
    print("-"*40)
    print("  BERT (Bio_ClinicalBERT) training is optional.")
    print("  It requires ~1-2 hours on GPU or 4-6 hours on CPU.")
    print("  The LSTM model above gives good results for local testing.")
    print("  To train BERT when ready:")
    print("    bert = BERTDiagnosisClassifier()")
    print("    bert.train(texts, labels, epochs=3)")

    print("\n" + "="*60)
    print("  NLP PIPELINE COMPLETE")
    print("="*60)
    print(f"  ✓ Text preprocessor  — {len(notes_df):,} notes cleaned")
    print(f"  ✓ LSTM classifier    — trained on {len(notes_df):,} notes")
    print(f"  ✓ Sentiment analyser — {len(feedback_samples)} feedback samples processed")
    print(f"  ✓ LSTM model saved   → models/nlp/lstm_model.pt")
    print("="*60)
