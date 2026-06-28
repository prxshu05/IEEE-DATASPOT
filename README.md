# 🔍 VeriLens AI
## Explainable Trust Verification System
**IEEE DataPort Hackathon 2026 — Machine Learning & AI Track**
**Problem Statement: Trustworthy Information Verification**

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model on your dataset
```bash
# Adapt your IEEE DataPort dataset first
python utils/dataset_adapter.py --input data/your_dataset.csv \
    --output data/train_standard.csv \
    --combine_cols title text

# Train
python train_model.py --train data/train_standard.csv \
    --epochs 3 --batch_size 16
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
verilens_ai/
│
├── app.py                        ← Streamlit Dashboard (Module 6)
├── train_model.py                ← Training entry point
├── requirements.txt
│
├── modules/
│   ├── input_handler.py          ← Module 1: Accept text / image
│   ├── ocr_extractor.py          ← Module 2: EasyOCR pipeline
│   ├── nlp_pipeline.py           ← Module 3: Text preprocessing
│   ├── model.py                  ← Module 4: DistilBERT classifier
│   └── trust_score.py            ← Module 5: Explainable Trust Score
│
├── utils/
│   ├── dataset_adapter.py        ← Convert IEEE DataPort datasets
│   └── evaluation.py             ← Metrics & confusion matrix
│
├── models/
│   └── fake_news_model/          ← Fine-tuned model saved here
│
└── data/                         ← Place your CSV datasets here
```

---

## 🧠 How It Works

### Pipeline
```
User Input (Text / Screenshot)
        ↓
[Module 1] Input Handler
        ↓
[Module 2] OCR (if image)
        ↓
[Module 3] NLP Preprocessing
        ↓
[Module 4] DistilBERT Classifier → Fake/Real + Probability
        ↓
[Module 5] Trust Score Engine
           - Model Confidence   (40%)
           - Language Quality   (20%)
           - Clickbait Score    (15%)
           - Emotional Language (10%)
           - Credibility Markers(10%)
           - Consistency        (5%)
        ↓
[Module 6] Streamlit Dashboard → Trust Score + Reasons + Charts
```

### Trust Score Interpretation
| Score | Verdict |
|-------|---------|
| 75–100 | ✅ Likely True |
| 55–74  | 🔍 Uncertain |
| 35–54  | ⚠️ Likely Misleading |
| 0–34   | ❌ Likely Fake |

---

## 📊 Dataset Adaptation

The `utils/dataset_adapter.py` supports:
- **WELFake** (title + text + label columns)
- **LIAR** (TSV format, 6-class → binary)
- **FakeNewsNet** (title/text/label)
- **Any generic CSV** (configurable column names)

```python
from utils.dataset_adapter import auto_detect_and_load

df = auto_detect_and_load("data/my_dataset.csv")
# Returns standard DataFrame: columns = ["text", "label"]
# label: 0 = REAL, 1 = FAKE
```

---

## 🏆 USP: Explainable Trust Score

Unlike other fake news detectors that only output **Fake / Real**, VeriLens AI provides:
- A **0–100 Trust Score**
- **Human-readable reasons** (clickbait detected, emotional language, low sourcing, etc.)
- **Component breakdown chart** showing what drove the score
- **Actionable recommendations** for the user

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| ML Model | HuggingFace Transformers (DistilBERT) |
| Training | PyTorch + HuggingFace Trainer |
| OCR | EasyOCR |
| Dashboard | Streamlit |
| Charts | Plotly |
| NLP | NLTK + TextBlob |

---

## 📈 Evaluation

```bash
python utils/evaluation.py --csv data/test.csv --text_col text --label_col label
```

---

## 🔮 Future Scope
- Claim-level verification
- Reverse image search
- AI-generated image detection (deepfake)
- Multilingual support
- Browser extension / WhatsApp bot
- Real-time news stream monitoring
