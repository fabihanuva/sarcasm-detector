"""
sarcasm_model.py
----------------
Core NLP pipeline for sarcasm detection.
Handles training, persistence, and real-time inference.
"""

import os
import re
import string
import joblib
import numpy as np
from typing import List, Tuple, Dict

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── NLTK with graceful fallback ────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    # Minimal built-in stopword list as fallback
    STOP_WORDS = {
        "i","me","my","we","our","you","your","he","she","it","they","their",
        "this","that","these","those","is","are","was","were","be","been",
        "being","have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","need","a","an","the","and",
        "but","or","nor","for","so","yet","at","by","in","of","on","to","up",
        "as","if","with","from","into","about","than","then","when","where",
    }

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "saved")
MODEL_PATH = os.path.join(MODEL_DIR, "sarcasm_pipeline.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

# Sarcasm-specific linguistic markers
SARCASM_MARKERS = [
    r"\boh sure\b", r"\byeah right\b", r"\bof course\b", r"\bright\.\.\.",
    r"\bsooo?\b", r"\btotally\b", r"\babsolutely\b", r"\bwow(,| )great\b",
    r"\bgreat job\b", r"\bwell done\b", r"!\s*!", r"\.\.\.",
    r"\bjust (what|perfect|wonderful|amazing)\b",
    r"\bwhat a (surprise|shock|shocker|genius|genius)\b",
    r"\bnot like\b", r"\bno (way|kidding|duh)\b",
    r"\bclearly\b", r"\bobviously\b",
]
MARKER_RE = re.compile("|".join(SARCASM_MARKERS), re.IGNORECASE)


def preprocess_text(text: str) -> str:
    """
    Clean and normalise raw text before vectorisation.
    Preserves sarcasm-relevant punctuation patterns as tokens.
    """
    text = text.lower()
    # Encode repeated punctuation as special tokens
    text = re.sub(r"!{2,}", " MULTI_EXCLAIM ", text)
    text = re.sub(r"\?{2,}", " MULTI_QUESTION ", text)
    text = re.sub(r"\.{3,}", " ELLIPSIS ", text)
    # Remove URLs and mentions
    text = re.sub(r"http\S+|www\.\S+|@\w+", " ", text)
    # Remove numbers (low signal for sarcasm)
    text = re.sub(r"\d+", " ", text)
    # Strip remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenise and remove stopwords
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════
# 2. TRAINING DATA  (built-in seed corpus — swap for real dataset)
# ══════════════════════════════════════════════════════════════════════════

SEED_DATA: List[Tuple[str, int]] = [
    # ── Sarcastic (label=1) ──────────────────────────────────────────────
    ("Oh great, another Monday. Just what I needed.", 1),
    ("Yeah right, because that always works perfectly.", 1),
    ("Wow, what a surprise! Nobody could have seen that coming.", 1),
    ("Oh sure, I totally believe everything you say.", 1),
    ("Because working overtime unpaid is everyone's dream, right?", 1),
    ("Oh fantastic, the meeting that could have been an email.", 1),
    ("Brilliant idea, why didn't I think of that? Oh wait...", 1),
    ("Oh yes, please tell me more about how you're always right.", 1),
    ("What a genius plan! Totally foolproof.", 1),
    ("Oh clearly that was the best decision ever made.", 1),
    ("Because obviously the customer is never wrong, right?", 1),
    ("Oh wow, what a groundbreaking discovery.", 1),
    ("Sure, because that's totally how science works.", 1),
    ("Oh no worries, I love waiting two hours for nothing.", 1),
    ("What a lovely surprise... not.", 1),
    ("Yeah, because traffic jams are so much fun.", 1),
    ("Oh wonderful, it's raining on my wedding day.", 1),
    ("Totally agree, this is the best code I've ever seen.", 1),
    ("Oh absolutely, I live to serve people who don't say thanks.", 1),
    ("Wow, another bug in production. Super exciting.", 1),
    ("Sure, let's add more features right before the deadline.", 1),
    ("Oh great, the app crashed again. My favourite thing.", 1),
    ("Yeah, because staying up all night is good for productivity.", 1),
    ("Oh look, another email marked urgent that isn't.", 1),
    ("Wow, what a revolutionary concept — actually doing your job.", 1),
    ("Oh no, I'm not frustrated at all. Everything is fine.", 1),
    ("Because apparently common sense is too much to ask for.", 1),
    ("Oh perfect, yet another pointless update to the software.", 1),
    ("Right, so the meeting is mandatory but the action items are optional.", 1),
    ("Wow, thanks for the constructive feedback of 'make it better'.", 1),

    # ── Sincere (label=0) ────────────────────────────────────────────────
    ("I really enjoyed the presentation today.", 0),
    ("The sunset was beautiful this evening.", 0),
    ("I am very grateful for your help with the project.", 0),
    ("The new library makes it much easier to handle requests.", 0),
    ("We need to fix this bug before the release.", 0),
    ("I think we should reconsider the timeline.", 0),
    ("The team did an amazing job under pressure.", 0),
    ("Please send me the report by Friday.", 0),
    ("I appreciate your patience during the outage.", 0),
    ("The database migration completed successfully.", 0),
    ("Can we schedule a call to discuss the requirements?", 0),
    ("I learned a lot from the training session.", 0),
    ("The weather forecast says it will rain tomorrow.", 0),
    ("She passed her exams with flying colours.", 0),
    ("I need some time to think about this decision.", 0),
    ("The coffee here is genuinely excellent.", 0),
    ("He worked hard and finally got the promotion.", 0),
    ("The new feature has improved user retention by 20 percent.", 0),
    ("I would like to understand the root cause of this issue.", 0),
    ("Thank you for staying late to help us finish.", 0),
    ("The architecture diagram helped clarify the system design.", 0),
    ("I found the book really thought-provoking.", 0),
    ("She genuinely cares about her team's wellbeing.", 0),
    ("The children were excited about the field trip.", 0),
    ("We successfully deployed the update with no downtime.", 0),
    ("I am looking forward to the conference next month.", 0),
    ("The documentation is clear and well structured.", 0),
    ("He sincerely apologised for the mistake.", 0),
    ("The new process has reduced errors significantly.", 0),
    ("I honestly think this is the right approach.", 0),
]


def get_training_data():
    """Return texts and labels, preferring an external CSV if present."""
    csv_path = os.path.join(os.path.dirname(__file__), "../../data/training_data.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df["text"].tolist(), df["label"].tolist()
    texts  = [t for t, _ in SEED_DATA]
    labels = [l for _, l in SEED_DATA]
    return texts, labels


# ══════════════════════════════════════════════════════════════════════════
# 3. PIPELINE CONSTRUCTION & TRAINING
# ══════════════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline:
      TF-IDF vectoriser  →  Calibrated LinearSVC
    LinearSVC is fast and strong on text; CalibratedClassifierCV
    adds probability estimates (needed for the Sarcasm Score %).
    """
    tfidf = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 3),        # unigrams, bigrams, trigrams
        max_features=20_000,
        sublinear_tf=True,         # apply log(tf) scaling
        min_df=1,
    )
    svc = CalibratedClassifierCV(
        LinearSVC(max_iter=2000, C=1.0),
        cv=3,
    )
    return Pipeline([("tfidf", tfidf), ("clf", svc)])


def train_model(force: bool = False) -> Pipeline:
    """
    Train (or load cached) model.
    Pass force=True to retrain even if a saved model exists.
    """
    if not force and os.path.exists(MODEL_PATH):
        print("[SarcasmModel] Loading cached model …")
        return joblib.load(MODEL_PATH)

    print("[SarcasmModel] Training new model …")
    texts, labels = get_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred,
                                target_names=["Sincere", "Sarcastic"]))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[SarcasmModel] Saved to {MODEL_PATH}")
    return pipeline


# ══════════════════════════════════════════════════════════════════════════
# 4. INFERENCE
# ══════════════════════════════════════════════════════════════════════════

# Singleton – loaded once per process
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = train_model()
    return _pipeline


def analyse_text(text: str) -> Dict:
    """
    Run sarcasm detection on a single piece of text.

    Returns
    -------
    dict with keys:
        score       – float 0-100  (probability of sarcasm)
        label       – "Sarcastic" | "Sincere"
        risk        – "high" | "low"
        highlights  – list of (word, weight) tuples for top contributing terms
        markers     – list of sarcasm-marker phrases found in raw text
    """
    pipeline  = get_pipeline()
    proba     = pipeline.predict_proba([text])[0]
    sarcasm_p = float(proba[1])              # probability class=1 (Sarcastic)
    score     = round(sarcasm_p * 100, 1)
    label     = "Sarcastic" if sarcasm_p >= 0.5 else "Sincere"
    risk      = "high" if sarcasm_p >= 0.5 else "low"

    highlights = _extract_highlights(text, pipeline, top_n=8)
    markers    = [m.group() for m in MARKER_RE.finditer(text)]

    return {
        "score":      score,
        "label":      label,
        "risk":       risk,
        "highlights": highlights,
        "markers":    list(set(markers)),
    }


def _extract_highlights(text: str, pipeline: Pipeline, top_n: int = 8
                        ) -> List[Dict]:
    """
    Identify words in `text` that most influenced the sarcastic classification.
    Uses TF-IDF feature weights × SVC coefficients.
    """
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        clf_cal    = pipeline.named_steps["clf"]
        coefs = [cal.estimator.coef_[0] for cal in clf_cal.calibrated_classifiers_ if hasattr(cal.estimator, "coef_")]
        if not coefs:
            return []
        coef = np.mean(coefs, axis=0)

        vec = vectorizer.transform([text])
        feature_names = np.array(vectorizer.get_feature_names_out())


        # Non-zero indices in this document's TF-IDF vector
        nz_indices = vec.nonzero()[1]
        scores = []
        for idx in nz_indices:
            word  = feature_names[idx]
            tfidf_val = vec[0, idx]
            contribution = float(tfidf_val * coef[idx])
            if contribution > 0:                  # positive = pushes toward sarcastic
                scores.append({"word": word, "weight": round(contribution, 4)})

        scores.sort(key=lambda x: x["weight"], reverse=True)
        return scores[:top_n]
    except Exception as e:
        print(f"[Highlights] Warning: {e}")
        return []


def analyse_bulk(texts: List[str]) -> List[Dict]:
    """Run analyse_text on a list of texts (file upload)."""
    return [analyse_text(t) for t in texts if t.strip()]
