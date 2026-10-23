"""Flask application factory and central configuration for ResumeFit."""

from __future__ import annotations

import logging
import secrets
from typing import Any

from flask import Flask, render_template, request, session

from resumefit.api.routes import api_bp, make_error_response
from resumefit.config import get_config
from resumefit.limiter import limiter
from resumefit.logging import setup_logging
from resumefit.web.routes import web_bp

logger = logging.getLogger(__name__)


def generate_csrf_token() -> str:
    """Generate or retrieve a cryptographically secure session-bound CSRF token."""
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(24)
    return str(session["_csrf_token"])


def warmup_models() -> None:
    """Pre-warm SentenceTransformer inference graph to eliminate cold-start user latency."""
    try:
        from core.matching.semantic import get_embedding_model

        model = get_embedding_model()
        model.encode(["system model warmup initialization query"], show_progress_bar=False)
        logger.info("SentenceTransformer model pre-warmed successfully.")
    except Exception as exc:
        logger.debug("Model warmup bypassed: %s", exc)


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure an instance of the Flask application."""
    config = get_config()
    setup_logging()

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    app.config["SECRET_KEY"] = config.app.secret_key
    app.config["MAX_CONTENT_LENGTH"] = config.ingestion.max_file_size_mb * 1024 * 1024

    if test_config:
        app.config.update(test_config)

    # Initialize rate limiter
    limiter.init_app(app)

    # Register CSRF template helper
    @app.context_processor
    def inject_csrf() -> dict[str, Any]:
        return {"csrf_token": generate_csrf_token}

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(web_bp)

    # Pre-warm embedding model unless running in fast unit test mode
    if not app.config.get("TESTING", False):
        warmup_models()

    # Central error handlers
    @app.errorhandler(400)
    def bad_request(e: Any) -> Any:
        if request.path.startswith("/api") or request.is_json:
            return make_error_response("BAD_REQUEST", str(e), 400)
        return render_template("error.html", code=400, message=str(e)), 400

    @app.errorhandler(404)
    def not_found(_e: Any) -> Any:
        if request.path.startswith("/api") or request.is_json:
            return make_error_response("NOT_FOUND", "The requested resource was not found.", 404)
        return render_template("error.html", code=404, message="Page or report not found."), 404

    @app.errorhandler(422)
    def unprocessable_entity(e: Any) -> Any:
        if request.path.startswith("/api") or request.is_json:
            return make_error_response("UNPROCESSABLE_ENTITY", str(e), 422)
        return render_template("error.html", code=422, message=str(e)), 422

    @app.errorhandler(429)
    def ratelimit_handler(e: Any) -> Any:
        msg = f"Rate limit exceeded: {e.description}"
        if request.path.startswith("/api") or request.is_json:
            return make_error_response("RATE_LIMIT_EXCEEDED", msg, 429)
        return render_template("error.html", code=429, message=msg), 429

    @app.errorhandler(500)
    def internal_server_error(e: Any) -> Any:
        logger.exception("Internal 500 error: %s", e)
        if request.path.startswith("/api") or request.is_json:
            return make_error_response("INTERNAL_SERVER_ERROR", "Internal server error.", 500)
        return render_template("error.html", code=500, message="An internal server error occurred."), 500

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="127.0.0.1", port=5000, debug=True)
