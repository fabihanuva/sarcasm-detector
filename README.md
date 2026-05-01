# 🎭 Sarcasm Detector

A full-stack sarcasm detection web application powered by NLP/ML.

---

## Folder Structure

```
sarcasm_detector/
│
├── run.py                          ← Entry point  (python run.py)
├── requirements.txt
│
├── app/
│   ├── __init__.py                 ← Flask factory (db init, blueprint register, model warm-up)
│   ├── routes.py                   ← All URL routes (analyse, result, history, export, API)
│   │
│   ├── models/
│   │   ├── sarcasm_model.py        ← NLP pipeline (preprocess → TF-IDF → LinearSVC → persist)
│   │   └── db_models.py            ← SQLAlchemy ORM (AnalysisSession, AnalysisRecord)
│   │
│   ├── utils/
│   │   ├── file_parser.py          ← .txt / .docx / .csv text extraction
│   │   └── exporter.py             ← PDF (ReportLab) and plain-text report generation
│   │
│   ├── templates/
│   │   ├── base.html               ← Shared nav, styles, flash messages
│   │   ├── index.html              ← Input dashboard (text paste + file upload tabs)
│   │   ├── result.html             ← Single-text result (score gauge, highlights, markers)
│   │   ├── bulk_result.html        ← File-upload bulk results table
│   │   └── history.html            ← Session history log + export actions
│   │
│   └── static/                     ← (place custom CSS/JS/images here if needed)
│
├── data/
│   └── training_data.csv           ← (optional) Custom training data: columns text,label
│
└── instance/
    └── sarcasm.db                  ← SQLite database (auto-created on first run)
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app  (model trains automatically on first launch)
python run.py
```

Visit **http://localhost:5000**

---

## How It Works: Frontend ↔ Backend Connection

```
User types text  →  POST /analyse  →  routes.py
                                           │
                    ┌──────────────────────┘
                    │
                    ▼
          sarcasm_model.analyse_text(text)
                    │
          ┌─────────┴──────────┐
          │  TF-IDF vectoriser │  (30k n-gram features, 1–3-grams)
          └─────────┬──────────┘
                    │ sparse matrix
          ┌─────────┴──────────┐
          │  CalibratedLinearSVC│  (probability calibration via CV)
          └─────────┬──────────┘
                    │
          {score, label, risk, highlights, markers}
                    │
          routes.py → DB write → redirect → result.html
```

**Highlighting pipeline** (client-side):
1. Server sends `highlights` (top TF-IDF × coefficient terms) as JSON in the template.
2. JavaScript regex-replaces matching words with `<mark class="sarcasm-mark">`.

---

## ML Pipeline Details

| Component | Choice | Reason |
|---|---|---|
| Vectoriser | TF-IDF (1–3-grams, 20k features) | Captures "oh great", "yeah right" as bigrams |
| Classifier | LinearSVC | Fast, strong on sparse text features |
| Calibration | CalibratedClassifierCV (cv=3) | Converts SVM margins → probabilities (0–100%) |
| Persistence | joblib | Fast binary serialisation of sklearn Pipeline |
| Preprocessing | Custom (regex + stopword removal) | Encodes `!!!` → MULTI_EXCLAIM token |

### Bring Your Own Dataset
Drop a `data/training_data.csv` with columns `text` (str) and `label` (0=sincere, 1=sarcastic).
Delete `app/models/saved/sarcasm_pipeline.joblib` and restart — the model will retrain on your data.

Popular open datasets:
- **News Headlines Sarcasm** (Rishabh Misra, ~28k headlines) — Kaggle
- **Reddit SARC corpus** (~1.3M comments)

---

## API Reference

### `POST /api/analyse`
```json
// Request
{ "text": "Oh great, another outage." }

// Response
{
  "score": 94.2,
  "label": "Sarcastic",
  "risk": "high",
  "highlights": [{"word": "oh", "weight": 0.156}, ...],
  "markers": ["oh great"]
}
```

---

## Feature Checklist

- [x] TF-IDF + LinearSVC pipeline with probability calibration
- [x] Custom preprocessing (special tokens for `!!!`, `...`, etc.)
- [x] Sarcasm-marker regex dictionary
- [x] Model persistence via joblib (no retraining on restart)
- [x] Text-paste input with character counter
- [x] File upload (.txt, .docx, .csv) with drag-and-drop
- [x] Single-text result page with animated score gauge
- [x] Word highlighting (TF-IDF × SVC coefficient weights)
- [x] Bulk results table for file uploads
- [x] SQLite history log per session (UUID cookie)
- [x] Export history as PDF (ReportLab) or plain text
- [x] Clear history action
- [x] JSON API endpoint (`/api/analyse`)
- [x] Responsive dark-mode UI (Bebas Neue + Space Mono)
