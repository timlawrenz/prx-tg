"""Sentry SDK initialization for production training.

Centralizes all Sentry configuration: error monitoring, tracing,
continuous profiling, and structured logs.

Usage:
    from .sentry_setup import init_sentry
    init_sentry()  # Call once at the top of main()
"""

import logging
import os
import subprocess

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration


log = logging.getLogger(__name__)


def _get_git_release() -> str | None:
    """Return a release string from the current git commit, or None."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{sha[:12]}{'-dirty' if dirty else ''}"
    except Exception:
        return None


def init_sentry() -> None:
    """Initialize Sentry SDK with tracing, profiling, and structured logs.

    Reads configuration from environment variables:
        SENTRY_DSN          – required (loaded from .env via python-dotenv)
        SENTRY_ENVIRONMENT  – optional, defaults to "development"
    """
    # Load .env file so SENTRY_DSN is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; rely on env being set externally

    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        log.warning("SENTRY_DSN not set – Sentry disabled")
        return

    environment = os.environ.get("SENTRY_ENVIRONMENT", "development")
    release = _get_git_release()

    # Configure Python logging so the LoggingIntegration can capture it
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        send_default_pii=False,

        # --- Tracing ---
        traces_sample_rate=1.0,

        # --- Continuous Profiling (trace lifecycle) ---
        profile_session_sample_rate=1.0,
        profile_lifecycle="trace",

        # --- Structured Logs ---
        enable_logs=True,

        # --- Logging integration ---
        integrations=[
            LoggingIntegration(
                sentry_logs_level=logging.INFO,
                level=logging.INFO,
                event_level=logging.ERROR,
            ),
        ],
    )

    log.info(
        "Sentry initialized (env=%s, release=%s, tracing=on, profiling=on, logs=on)",
        environment,
        release or "auto",
    )
