🎭 Sarcasm Detector
An intelligent full-stack web application that detects sarcasm in text using a dual-engine NLP and Machine Learning pipeline. Built with Python, Flask, and Scikit-Learn — trained on 26,000+ real-world news headlines.

🚀 Live Demo
Run it locally and open http://127.0.0.1:5001 in your browser.

✨ Features

Sarcasm Score — probability percentage from 0% to 100%
Risk Level Indicator — 🔴 Red for Sarcastic, 🟢 Green for Sincere
Phrase Highlighting — highlights the exact words that triggered the sarcastic classification
File Upload Support — paste raw text or upload .txt, .docx, or .csv files for bulk detection
Analysis History — every analysis is saved per session to a SQLite database
Export Reports — download your full history as a PDF or plain text file
REST API — JSON endpoint for programmatic access
Session Management — each user has their own private history log


📁 Project Structure
sarcasm_detector/
│
├── run.py                        ← Entry point — launches the Flask server
├── requirements.txt              ← All Python dependencies
├── setup.sh                      ← One-command setup script for new machines
│
├── app/
│   ├── __init__.py               ← Flask app factory (registers DB, blueprint, warms up model)
│   ├── routes.py                 ← All URL routes (/analyse, /result, /history, /export, /api)
│   │
│   ├── models/
│   │   ├── sarcasm_model.py      ← Core NLP + ML pipeline (TF-IDF → LinearSVC → Semantic Engine)
│   │   └── db_models.py          ← SQLAlchemy ORM models (AnalysisSession, AnalysisRecord)
│   │
│   ├── utils/
│   │   ├── file_parser.py        ← Extracts text from .txt / .docx / .csv uploads
│   │   └── exporter.py           ← Generates PDF (ReportLab) and plain-text reports
│   │
│   └── templates/
│       ├── base.html             ← Shared layout (nav, styles, flash messages)
│       ├── index.html            ← Input dashboard (text paste + drag-and-drop file upload)
│       ├── result.html           ← Single analysis result (score gauge, highlights, markers)
│       ├── bulk_result.html      ← Bulk results table for file uploads
│       └── history.html          ← Session history log with export options
│
├── data/
│   └── training_data.csv         ← Training dataset (26,709 labelled headlines)
│
└── instance/
    └── sarcasm.db                ← SQLite database (auto-created on first run)

⚙️ Installation & Setup
1. Clone the Repository
bashgit clone https://github.com/fabihanuva/sarcasm-detector.git
cd sarcasm-detector
2. Create a Virtual Environment
bash# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
3. Install Dependencies
bashpip install -r requirements.txt
4. Run the App
bashpython3 run.py
Open your browser and go to → http://127.0.0.1:5001

The model trains automatically on first launch. This takes about 30–60 seconds. Subsequent launches load the saved model instantly.


🧠 How It Works
App Flow
User inputs text or uploads file
           │
           ▼
    POST /analyse  (routes.py)
           │
           ▼
  sarcasm_model.analyse_text()
           │
    ┌──────┴───────────────────────┐
    │                              │
    ▼                              ▼
Layer 1: TF-IDF + LinearSVC    Layer 2: Semantic Contrast Engine
(Statistical ML baseline)      (Emotion vs Context gap detection)
    │                              │
    └──────────────┬───────────────┘
                   │
           Layer 3: Sarcasm Marker Detection
           (Phrase patterns: "oh great", "yeah right")
                   │
                   ▼
        { score, label, risk, highlights, markers }
                   │
                   ▼
        Saved to SQLite → result.html
Preprocessing

Converts text to lowercase
Encodes !!! → MULTI_EXCLAIM, ... → ELLIPSIS as special tokens
Removes URLs, numbers, and punctuation
Removes stopwords while keeping sarcasm-signal words (oh, wow, not, so)

Feature Engineering

TF-IDF Vectorisation — 1 to 3-gram features (40,000 features max)
Captures multi-word sarcasm phrases like "oh great", "yeah right", "of course"

ML Model

LinearSVC wrapped in CalibratedClassifierCV
Calibration converts raw SVM margins into proper probabilities (0–100%)
Trained on the News Headlines Sarcasm Dataset (26,709 headlines)

Semantic Contrast Engine
Detects the meaning gap between expressed emotion and actual context — something pure word-counting cannot do.
InputDetected PatternResult"I love it when my computer crashes"love (positive emotion) + crashes (negative context)🔴 Sarcastic"I love spending time with my family"love (positive emotion) + no negative context🟢 Sincere"Oh great, another Monday"Explicit sarcasm marker🔴 Sarcastic
Word Highlighting
Top contributing words are identified by multiplying each word's TF-IDF weight by its LinearSVC coefficient. Words with the highest score are highlighted in red on the result page.

📊 ML Pipeline Details
ComponentChoiceReasonVectoriserTF-IDF (1–3-grams, 40k features)Captures multi-word sarcasm phrasesClassifierLinearSVCFast and strong on sparse high-dimensional textCalibrationCalibratedClassifierCV (cv=5)Converts SVM scores into 0–100% probabilitiesPersistencejoblibFast binary model save/load (no retraining on restart)PreprocessingCustom regex pipelineEncodes punctuation patterns as meaningful tokens

📂 Dataset
This project uses the News Headlines Sarcasm Dataset by Rishabh Misra.
PropertyValueTotal samples26,709 headlinesSarcastic~13,000 (from The Onion)Sincere~13,700 (from HuffPost)SourceKaggle
To retrain the model on the dataset:
bashrm -f app/models/saved/sarcasm_pipeline.joblib
python3 run.py

🔌 API Reference
POST /api/analyse
Analyse a single piece of text programmatically.
Request
json{
  "text": "Oh great, another Monday. Just what I needed."
}
Response
json{
  "score": 99.5,
  "label": "Sarcastic",
  "risk": "high",
  "highlights": [
    { "word": "oh",      "weight": 0.156 },
    { "word": "another", "weight": 0.089 },
    { "word": "great",   "weight": 0.065 }
  ],
  "markers": ["oh great", "just what i needed"]
}

✅ Feature Checklist

 Three-layer hybrid detection (ML + Semantic + Markers)
 TF-IDF + LinearSVC with probability calibration
 Semantic Contrast Engine (understands meaning, not just words)
 Sarcasm marker regex dictionary (30+ patterns)
 Model persistence via joblib (no retraining on restart)
 Text paste input with live character counter
 File upload (.txt, .docx, .csv) with drag-and-drop
 Animated sarcasm score gauge
 Word and phrase highlighting on result page
 Bulk results table for file uploads
 SQLite history log per session (UUID cookie)
 Export history as PDF (ReportLab) or plain text
 Clear history action
 JSON REST API endpoint
 Dark-mode responsive UI


🛠️ Tech Stack
LayerTechnologyWeb FrameworkFlask 3.0Machine LearningScikit-Learn (LinearSVC, TF-IDF)DatabaseSQLite via SQLAlchemyPDF ExportReportLabFile Parsingpython-docx, csvNLP UtilitiesNLTKModel PersistencejoblibFrontendJinja2 Templates, Vanilla JS

👩‍💻 Running Every Time
bashcd sarcasm_detector
source .venv/bin/activate
python3 run.py
Then open → http://127.0.0.1:5001
Press Ctrl+C to stop the server.

📄 License
This project was built for academic purposes.
Dataset credit: Rishabh Misra — News Headlines Dataset for Sarcasm Detection, 2019.