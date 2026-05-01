"""
app/models/db_models.py
-----------------------
SQLAlchemy ORM models for analysis history.
"""

from datetime import datetime
from .. import db


class AnalysisSession(db.Model):
    """Groups multiple analyses under one user session."""
    __tablename__ = "analysis_session"

    id          = db.Column(db.Integer, primary_key=True)
    session_key = db.Column(db.String(64), nullable=False, index=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    analyses    = db.relationship("AnalysisRecord", backref="session",
                                  lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":         self.id,
            "session_key": self.session_key,
            "created_at": self.created_at.isoformat(),
            "count":      len(self.analyses),
        }


class AnalysisRecord(db.Model):
    """Single text-analysis result."""
    __tablename__ = "analysis_record"

    id          = db.Column(db.Integer, primary_key=True)
    session_id  = db.Column(db.Integer, db.ForeignKey("analysis_session.id"),
                            nullable=False)
    input_text  = db.Column(db.Text, nullable=False)
    score       = db.Column(db.Float, nullable=False)         # 0–100
    label       = db.Column(db.String(16), nullable=False)    # Sarcastic / Sincere
    risk        = db.Column(db.String(8),  nullable=False)    # high / low
    highlights  = db.Column(db.Text)                          # JSON string
    markers     = db.Column(db.Text)                          # JSON string
    source      = db.Column(db.String(32), default="text")    # text / file
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        return {
            "id":         self.id,
            "input_text": self.input_text,
            "score":      self.score,
            "label":      self.label,
            "risk":       self.risk,
            "highlights": json.loads(self.highlights or "[]"),
            "markers":    json.loads(self.markers or "[]"),
            "source":     self.source,
            "created_at": self.created_at.isoformat(),
        }
