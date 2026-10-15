"""CSRF protection and shared SQLite limits for authentication attempts."""
import hashlib
import sqlite3
import time
from contextlib import closing

from flask import jsonify, request, session
from flask_wtf.csrf import CSRFError, CSRFProtect, generate_csrf


def consume_auth_attempt(database_path, action, address, username="", now=None):
    """Atomically charge both IP and account/IP windows across web workers."""
    now = time.time() if now is None else now
    policies = [(f"{action}:ip:{address}", 30 if action == "login" else 10)]
    if action == "login":
        policies.append((f"login:account:{address}:{username}", 5))
    with closing(sqlite3.connect(database_path, timeout=10)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS auth_limits "
                           "(key TEXT PRIMARY KEY, attempts INTEGER NOT NULL, expires REAL NOT NULL)")
        connection.execute("DELETE FROM auth_limits WHERE expires <= ?", (now,))
        entries = []
        for key, limit in policies:
            key = hashlib.sha256(key.encode("utf-8")).hexdigest()
            row = connection.execute("SELECT attempts, expires FROM auth_limits WHERE key=?", (key,)).fetchone()
            if row and row[0] >= limit:
                return max(1, int(row[1] - now + 1))
            entries.append((key, row))
        for key, row in entries:
            if row:
                connection.execute("UPDATE auth_limits SET attempts=attempts+1 WHERE key=?", (key,))
            else:
                connection.execute("INSERT INTO auth_limits VALUES (?, 1, ?)", (key, now + 300))
    return 0


def valid_password(password):
    return (8 <= len(password) <= 128 and any(c.isalpha() for c in password)
            and any(not c.isalpha() for c in password))


def configure_csrf(app):
    # Allow ops' authentication guard to produce 401/403 before CSRF validation.
    app.config["WTF_CSRF_CHECK_DEFAULT"] = False
    csrf = CSRFProtect(app)

    @app.before_request
    def protect_writes():
        if request.path.startswith("/ops/") and (
            session.get("logged_in") is not True or not session.get("username")
            or session.get("is_admin") is not True
        ):
            return None
        if app.config.get("WTF_CSRF_ENABLED", True):
            csrf.protect()

    @app.errorhandler(CSRFError)
    def csrf_error(error):
        return jsonify({"error": "CSRF validation failed", "message": "页面已过期，请刷新后重试"}), 400

    @app.get("/auth/csrf")
    def csrf_token():
        response = jsonify({"csrf_token": generate_csrf()})
        response.headers["Cache-Control"] = "no-store"
        return response
