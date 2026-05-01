"""
app/__init__.py  —  Flask Application Factory
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
import os

db = SQLAlchemy()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── Config ────────────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "sarcasm-dev-secret-2024")
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instance")
    os.makedirs(db_dir, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"sqlite:///{os.path.join(db_dir, 'sarcasm.db')}"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB upload limit

    # Flask-Session config
    app.config["SESSION_TYPE"] = "sqlalchemy"
    app.config["SESSION_SQLALCHEMY"] = db

    db.init_app(app)
    Session(app)

    with app.app_context():
        from .routes import main
        app.register_blueprint(main)
        db.create_all()

        # Warm up the model on first run (trains + saves if not cached)
        from .models.sarcasm_model import get_pipeline
        get_pipeline()

    return app
