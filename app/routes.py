"""
app/routes.py
-------------
All Flask routes for the Sarcasm Detector application.
"""

import json
import uuid
from datetime import datetime

from flask import (
    Blueprint, render_template, request, session,
    redirect, url_for, flash, jsonify, send_file, abort
)
import io

from . import db
from .models.db_models  import AnalysisSession, AnalysisRecord
from .models.sarcasm_model import analyse_text, analyse_bulk
from .utils.file_parser import parse_uploaded_file
from .utils.exporter    import export_as_text, export_as_pdf

main = Blueprint("main", __name__)


# ── Helper: get or create a session key ──────────────────────────────────
def get_session_key() -> str:
    if "session_key" not in session:
        session["session_key"] = str(uuid.uuid4())
    return session["session_key"]


def get_or_create_db_session(key: str) -> AnalysisSession:
    s = AnalysisSession.query.filter_by(session_key=key).first()
    if not s:
        s = AnalysisSession(session_key=key)
        db.session.add(s)
        db.session.commit()
    return s


# ════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════

@main.route("/")
def index():
    """Landing / input dashboard."""
    return render_template("index.html")


@main.route("/analyse", methods=["POST"])
def analyse():
    """
    Handle text paste OR file upload.
    Runs inference → stores to DB → redirects to result page.
    """
    key        = get_session_key()
    db_session = get_or_create_db_session(key)
    source     = "text"
    texts      = []

    # ── Determine input source ──────────────────────────────────────────
    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        try:
            texts  = parse_uploaded_file(f)
            source = "file"
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("main.index"))
    else:
        raw = request.form.get("text_input", "").strip()
        if not raw:
            flash("Please enter some text or upload a file.", "error")
            return redirect(url_for("main.index"))
        texts = [raw]

    if not texts:
        flash("No text found to analyse.", "error")
        return redirect(url_for("main.index"))

    # ── Run inference ───────────────────────────────────────────────────
    results  = analyse_bulk(texts)
    record_ids = []

    for text, result in zip(texts, results):
        record = AnalysisRecord(
            session_id = db_session.id,
            input_text = text[:4000],        # cap stored length
            score      = result["score"],
            label      = result["label"],
            risk       = result["risk"],
            highlights = json.dumps(result["highlights"]),
            markers    = json.dumps(result["markers"]),
            source     = source,
        )
        db.session.add(record)
        db.session.flush()
        record_ids.append(record.id)

    db.session.commit()

    # Single text → detail page; bulk → bulk results page
    if len(record_ids) == 1:
        return redirect(url_for("main.result", record_id=record_ids[0]))
    return redirect(url_for("main.bulk_result",
                            ids=",".join(map(str, record_ids))))


@main.route("/result/<int:record_id>")
def result(record_id):
    """Single-analysis result page."""
    record = AnalysisRecord.query.get_or_404(record_id)
    data   = record.to_dict()
    return render_template("result.html", record=data)


@main.route("/bulk-result")
def bulk_result():
    """Bulk-upload results page."""
    ids_str = request.args.get("ids", "")
    try:
        ids = [int(i) for i in ids_str.split(",") if i]
    except ValueError:
        abort(400)
    records = [r.to_dict() for r in
               AnalysisRecord.query.filter(AnalysisRecord.id.in_(ids)).all()]
    return render_template("bulk_result.html", records=records)


@main.route("/history")
def history():
    """User history log page."""
    key     = get_session_key()
    db_sess = AnalysisSession.query.filter_by(session_key=key).first()
    records = []
    if db_sess:
        records = [r.to_dict() for r in
                   AnalysisRecord.query
                   .filter_by(session_id=db_sess.id)
                   .order_by(AnalysisRecord.created_at.desc())
                   .limit(100).all()]
    return render_template("history.html", records=records)


@main.route("/export/<fmt>")
def export(fmt):
    """Download all session records as PDF or TXT."""
    key     = get_session_key()
    db_sess = AnalysisSession.query.filter_by(session_key=key).first()
    if not db_sess:
        flash("No history to export.", "error")
        return redirect(url_for("main.history"))

    records = [r.to_dict() for r in db_sess.analyses]
    ts      = datetime.utcnow().strftime("%Y%m%d_%H%M")

    if fmt == "pdf":
        data     = export_as_pdf(records)
        mimetype = "application/pdf"
        filename = f"sarcasm_report_{ts}.pdf"
    else:
        data     = export_as_text(records)
        mimetype = "text/plain"
        filename = f"sarcasm_report_{ts}.txt"

    return send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )


@main.route("/api/analyse", methods=["POST"])
def api_analyse():
    """JSON API endpoint for programmatic access."""
    body = request.get_json(force=True)
    text = (body or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = analyse_text(text)
    return jsonify(result)


@main.route("/clear-history", methods=["POST"])
def clear_history():
    """Delete this session's history."""
    key     = get_session_key()
    db_sess = AnalysisSession.query.filter_by(session_key=key).first()
    if db_sess:
        db.session.delete(db_sess)
        db.session.commit()
    flash("History cleared.", "info")
    return redirect(url_for("main.history"))
