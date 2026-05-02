"""
sarcasm_model.py
----------------
Advanced Dual-Engine Sarcasm Detector.
Combines Statistical NLP (TF-IDF + LinearSVC) with
Semantic Contrast Analysis (Emotion vs Context gap detection).
"""

import os
import re
import string
import joblib
import numpy as np
from typing import List, Tuple, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# NLTK stopwords with graceful fallback
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","he","him","his","himself","she","her","hers","herself","it",
        "its","itself","they","them","their","theirs","themselves","what","which",
        "who","whom","this","that","these","those","am","is","are","was","were",
        "be","been","being","have","has","had","having","do","does","did","doing",
        "a","an","the","and","but","if","or","because","as","until","while","of",
        "at","by","for","with","about","against","between","into","through",
        "during","before","after","above","below","to","from","up","down","in",
        "out","on","off","over","under","again","further","then","once","here",
        "there","when","where","why","how","all","both","each","few","more",
        "most","other","some","such","no","nor","not","only","own","same","so",
        "than","too","very","s","t","can","will","just","don","should","now",
    }

# File paths
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_PATH = os.path.join(MODEL_DIR, "sarcasm_pipeline.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

SARCASM_MARKERS = [
    r"\boh[,\s]+(?:i\s+)?(?:really\s+|so\s+|just\s+)?(?:sure|great|fantastic|wonderful|brilliant|perfect|lovely|exciting|love|amazing|wow)\b",
    r"\byeah[,\s]+right\b",
    r"\bof\s+course\b",
    r"\bright\s*\.\.\.",
    r"\bsooo+\b",
    r"\btotally\b",
    r"\babsolutely\b",
    r"\bwow[,\s]+great\b",
    r"\bgreat\s+job\b",
    r"\bwell\s+done\b",
    r"!\s*!",
    r"\.{3,}",
    r"\bjust\s+(?:what|perfect|wonderful|amazing|what\s+I\s+needed|what\s+I\s+wanted)\b",
    r"\bwhat\s+a\s+(?:surprise|shock|shocker|genius|relief|treat|delight)\b",
    r"\bnot\s+like\b",
    r"\bno\s+(?:way|kidding|duh)\b",
    r"\bclearly\b",
    r"\bobviously\b",
    r"\bsurprise[,\s]+surprise\b",
    r"\bthanks\s+for\s+nothing\b",
    r"\bbig\s+deal\b",
    r"\bas\s+if\b",
    r"\bwho\s+would\s+have\s+thought\b",
    r"\bshocker\b",
    r"\bgroundbreaking\b",
    r"\bmy\s+fav(?:ou?rite)?\s+thing\b",
    r"\bcan'?t\s+wait\b",
    r"\bso\s+excited\b",
    r"\bnot\s+at\s+all\b",
    r"\beverything\s+is\s+fine\b",
]
MARKER_RE = re.compile("|".join(SARCASM_MARKERS), re.IGNORECASE)


def preprocess_text(text: str) -> str:
    """
    Normalise text for TF-IDF vectorisation.
    Keeps sarcasm-signal words that standard stopword lists would remove.
    """
    text = text.lower()
    text = re.sub(r"!{2,}", " MULTI_EXCLAIM ", text)
    text = re.sub(r"\?{2,}", " MULTI_QUESTION ", text)
    text = re.sub(r"\.{3,}", " ELLIPSIS ", text)
    text = re.sub(r"([!?.])", r" \1 ", text)
    text = re.sub(r"http\S+|www\.\S+|@\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Keep sarcasm-signal words even if they're stopwords
    keep = {"oh", "no", "so", "not", "too", "wow", "well", "just", "now"}
    tokens = [
        t for t in text.split()
        if (t not in STOP_WORDS or t in keep) and len(t) > 1
    ]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════
# 2. TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════

SEED_DATA: List[Tuple[str, int]] = [
    # Sarcastic
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
    ("Oh, I simply love it when my computer crashes right before a deadline.", 1),
    ("I really enjoy paying taxes. Best day of the year.", 1),
    ("What a wonderful day to be stuck in an elevator.", 1),
    ("I'm so excited to stay late at the office on a Friday night.", 1),
    ("Oh brilliant, another bill to pay. My favourite thing.", 1),
    ("Yeah, because being insulted is totally what I needed today.", 1),
    ("Because obviously the customer is never wrong, right?", 1),
    ("Oh wow, what a groundbreaking discovery. Who would have thought.", 1),
    ("Sure, because that's totally how science works.", 1),
    ("Oh no worries, I love waiting two hours for nothing.", 1),
    ("What a lovely surprise... not.", 1),
    ("Yeah, because traffic jams are so much fun.", 1),
    ("Oh wonderful, it's raining on my wedding day.", 1),
    ("Oh absolutely, I live to serve people who never say thanks.", 1),
    ("Wow, another bug in production. Super exciting stuff.", 1),
    ("Sure, let's add more features right before the deadline. Great idea.", 1),
    ("Oh great, the app crashed again. My favourite thing.", 1),
    ("Oh look, another email marked urgent that clearly isn't.", 1),
    ("Oh no, I'm not frustrated at all. Everything is fine.", 1),
    ("Because apparently common sense is too much to ask for.", 1),
    # Sincere
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
    ("The weather is genuinely beautiful today.", 0),
    ("She passed her exams with flying colours.", 0),
    ("I need some time to think about this decision.", 0),
    ("The coffee here is genuinely excellent.", 0),
    ("He worked hard and finally got the promotion he deserved.", 0),
    ("I would like to understand the root cause of this issue.", 0),
    ("Thank you for staying late to help us finish.", 0),
    ("She genuinely cares about her team's wellbeing.", 0),
    ("We successfully deployed the update with no downtime.", 0),
    ("I am looking forward to the conference next month.", 0),
    ("The documentation is clear and well structured.", 0),
    ("He sincerely apologised for the mistake.", 0),
    ("I honestly think this is the right approach for the team.", 0),
    ("I love spending time with my family on weekends.", 0),
    ("The children were genuinely excited about the field trip.", 0),
    ("I am so happy for her, she really deserved this opportunity.", 0),
    ("The weather forecast says it will rain tomorrow.", 0),
    ("I found the book really thought-provoking and well written.", 0),
]


def get_training_data() -> Tuple[List[str], List]:
    """Load from CSV if available, otherwise fall back to seed data."""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/training_data.csv"
    )
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"[SarcasmModel] Loaded {len(df):,} rows from training_data.csv")
        return df["text"].tolist(), df["label"].tolist()
    print("[SarcasmModel] No CSV found, using built-in seed data")
    return [t for t, _ in SEED_DATA], [l for _, l in SEED_DATA]


# ══════════════════════════════════════════════════════════════════════════
# 3. PIPELINE CONSTRUCTION & TRAINING
# ══════════════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """
    TF-IDF (1-3-grams, 40k features) + CalibratedLinearSVC.
    dual=False is required when n_samples > n_features (large datasets).
    CalibratedClassifierCV gives us predict_proba() for the 0-100% score.
    """
    tfidf = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 3),
        max_features=40_000,
        sublinear_tf=True,
        min_df=2,
    )
    svc = CalibratedClassifierCV(
        LinearSVC(max_iter=5000, C=0.5, dual=False),
        cv=5,
    )
    return Pipeline([("tfidf", tfidf), ("clf", svc)])


def train_model(force: bool = False) -> Pipeline:
    """Train and save, or load cached model. force=True retrains from scratch."""
    if not force and os.path.exists(MODEL_PATH):
        print("[SarcasmModel] Loading cached model ...")
        return joblib.load(MODEL_PATH)

    print("[SarcasmModel] Training new model ...")
    texts, labels = get_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Sincere", "Sarcastic"]))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[SarcasmModel] Model saved to {MODEL_PATH}")
    return pipeline


# ══════════════════════════════════════════════════════════════════════════
# 4. SEMANTIC CONTRAST ENGINE
#    Detects the "meaning gap" between expressed emotion and real context.
#    e.g. "I love it when my computer crashes" — love (positive emotion)
#    vs crashes (negative context) = sarcasm gap detected.
# ══════════════════════════════════════════════════════════════════════════

LEX_EMOTION_POS = {
    "love", "loving", "loved", "enjoy", "enjoyed", "enjoying", "happy",
    "glad", "relief", "excited", "exciting", "favorite", "favourite",
    "heaven", "joy", "brilliant", "wonderful", "fantastic", "amazing",
    "genius", "perfect", "perfectly", "excellent", "superb", "impressive",
    "spectacular", "outstanding", "thrilled", "delighted", "pleased",
    "great", "best", "incredible", "awesome",
}

LEX_CONTEXT_NEG = {
    "crash", "crashed", "crashes", "crashing", "late", "traffic", "jam",
    "stuck", "elevator", "taxes", "tax", "bill", "bills", "broken", "forgot",
    "forget", "forgotten", "mistake", "error", "fail", "failed", "failure",
    "waste", "wasted", "useless", "boring", "slow", "insult", "insulted",
    "unpaid", "lost", "losing", "meeting", "expensive", "bad", "terrible",
    "awful", "worst", "hard", "difficult", "deadline", "wait", "waiting",
    "nothing", "monday", "mondays", "rain", "cold", "problem", "issue",
    "bug", "outage", "delay", "cancelled", "denied", "rejected", "fired",
    "layoff", "debt", "overdraft", "missing",
}

LEX_INTENSIFIERS = {
    "totally", "absolutely", "clearly", "obviously", "simply", "really",
    "so", "just", "literally", "honestly", "genuinely", "truly",
}

NEG_PHRASES = [
    "don't work", "doesn't work", "not working", "no one", "not like",
    "too much", "can't stand", "hate when", "worst part", "nothing works",
    "never works", "always wrong", "always fails",
]


def get_semantic_contrast_score(text: str) -> float:
    """
    The Semantic Contrast Engine.

    Measures the gap between expressed positive emotion and negative context.
    This is what allows the model to understand MEANING rather than just
    matching individual words.

    Returns:
      positive float  -> push toward Sarcastic
      negative float  -> push toward Sincere (sincerity shield)
      0.0             -> neutral, let ML model decide
    """
    text_low     = text.lower()
    words        = set(preprocess_text(text).split())
    emotion_pos  = [w for w in words if w in LEX_EMOTION_POS]
    context_neg  = [w for w in words if w in LEX_CONTEXT_NEG]
    intensifiers = [w for w in words if w in LEX_INTENSIFIERS]
    has_neg_phrase = any(p in text_low for p in NEG_PHRASES)

    # Rule A: Positive emotion word + negative situation = sarcasm gap
    # "I love it when my computer crashes" → love + crashes = sarcastic
    if emotion_pos and (context_neg or has_neg_phrase):
        gap = 0.55
        gap += min(len(intensifiers), 3) * 0.08   # intensifiers amplify
        if text.rstrip().endswith("!"):
            gap += 0.08                            # exclamation seals it
        return round(gap, 3)

    # Rule B: Sincerity protection
    # Positive emotion + no negative context + no markers = genuine
    # "I love spending time with my family" → sincere, don't flag it
    if emotion_pos and not context_neg and not has_neg_phrase:
        if not MARKER_RE.search(text):
            return -0.55

    # Rule C: Pure negative ("I hate traffic") = sincere complaint, not sarcasm
    # Return 0 and let the ML baseline handle it
    return 0.0


# ══════════════════════════════════════════════════════════════════════════
# 5. INFERENCE
# ══════════════════════════════════════════════════════════════════════════

_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = train_model()
    return _pipeline


def analyse_text(text: str) -> Dict:
    """
    Three-layer hybrid sarcasm analysis:

    Layer 1 - ML Baseline : TF-IDF + LinearSVC probability score
    Layer 2 - Semantic Gap: Emotion vs Context contrast (meaning engine)
    Layer 3 - Marker Boost: Explicit sarcasm phrase pattern detection

    Returns dict: score (0-100), label, risk, highlights, markers
    """
    if not text or not text.strip():
        return {"score": 0.0, "label": "Sincere", "risk": "low",
                "highlights": [], "markers": []}

    pipeline  = get_pipeline()
    proba     = pipeline.predict_proba([text])[0]
    sarcasm_p = float(proba[1])          # Layer 1: raw ML probability

    # Layer 2: semantic contrast engine
    contrast_boost = get_semantic_contrast_score(text)
    sarcasm_p     += contrast_boost

    # Layer 3: explicit marker detection
    markers = list(set(m.group().lower() for m in MARKER_RE.finditer(text)))
    if markers and contrast_boost > -0.4:
        sarcasm_p += 0.20

    # Clamp to valid range
    sarcasm_p = max(0.01, min(0.995, sarcasm_p))

    score = round(sarcasm_p * 100, 1)
    label = "Sarcastic" if sarcasm_p >= 0.50 else "Sincere"
    risk  = "high"      if sarcasm_p >= 0.50 else "low"

    highlights = _extract_highlights(text, pipeline, top_n=8)

    return {
        "score":      score,
        "label":      label,
        "risk":       risk,
        "highlights": highlights,
        "markers":    markers,
    }


def _extract_highlights(text: str, pipeline: Pipeline, top_n: int = 8) -> List[Dict]:
    """
    Find words in the input that most pushed the ML model toward Sarcastic.
    Uses TF-IDF weight x averaged SVC coefficients across all CV folds.
    """
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        clf_cal    = pipeline.named_steps["clf"]
        coefs = [
            cal.estimator.coef_[0]
            for cal in clf_cal.calibrated_classifiers_
            if hasattr(cal.estimator, "coef_")
        ]
        if not coefs:
            return []
        coef = np.mean(coefs, axis=0)

        vec           = vectorizer.transform([text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        nz_indices    = vec.nonzero()[1]

        scores = []
        for idx in nz_indices:
            contribution = float(vec[0, idx] * coef[idx])
            if contribution > 0:
                scores.append({
                    "word":   feature_names[idx],
                    "weight": round(contribution, 4),
                })
        scores.sort(key=lambda x: x["weight"], reverse=True)
        return scores[:top_n]
    except Exception as e:
        print(f"[Highlights] Warning: {e}")
        return []


def analyse_bulk(texts: List[str]) -> List[Dict]:
    """Run analyse_text on every non-empty line (used for file uploads)."""
    return [analyse_text(t) for t in texts if t and t.strip()]
