"""Explicit production configuration and safe local server defaults."""
import os
import secrets


def application_config(environ=None):
    env = os.environ if environ is None else environ
    mode = env.get("BIM_WEB_ENV", "production")
    if mode not in {"production", "development"}:
        raise ValueError("BIM_WEB_ENV must be production or development")
    secret = env.get("SECRET_KEY", "").strip()
    if mode == "production" and (
        len(secret) < 32 or secret.startswith("replace-")
    ):
        raise ValueError("Production requires a fixed SECRET_KEY of at least 32 characters")
    max_upload_mb = int(env.get("MAX_UPLOAD_MB", "20"))
    if not 1 <= max_upload_mb <= 200:
        raise ValueError("MAX_UPLOAD_MB must be between 1 and 200")
    return {
        "SECRET_KEY": secret or secrets.token_hex(32),
        "SESSION_COOKIE_NAME": env.get("SESSION_COOKIE_NAME", "bim_web_session_v2"),
        "SESSION_COOKIE_SECURE": mode == "production",
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "MAX_CONTENT_LENGTH": max_upload_mb * 1024 * 1024,
        "BIM_WEB_ENV": mode,
    }


def local_run_options(environ=None):
    env = os.environ if environ is None else environ
    debug = env.get("BIM_WEB_DEBUG", "0").lower() in {"1", "true"}
    if debug and env.get("BIM_WEB_ENV") != "development":
        raise ValueError("Debugging requires BIM_WEB_ENV=development")
    port = int(env.get("PORT", "5000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    return {"host": env.get("BIM_WEB_HOST", "127.0.0.1"),
            "port": port, "debug": debug, "threaded": True}
