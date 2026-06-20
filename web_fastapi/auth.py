"""Session-based authentication — lightweight, no external dependencies."""

from __future__ import annotations

import json
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, status

from web_fastapi.user_store import User, get_user_by_username


# ── Session store (persisted to disk) ─────────────────────────────────────────

SESSION_EXPIRE_SECONDS = 24 * 3600  # 24 hours
_DATA_DIR = Path(__file__).parent.parent / "userdata"
_SESSIONS_FILE = _DATA_DIR / "sessions.json"
_session_store: dict[str, dict] = {}
_session_lock = threading.Lock()


def _init_sessions():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _SESSIONS_FILE.exists():
        try:
            with open(_SESSIONS_FILE) as f:
                raw = json.load(f)
            now = time.time()
            for token, data in list(raw.items()):
                if data.get("expires_at", 0) > now:
                    _session_store[token] = data
            logger.info(f"Loaded {len(_session_store)} active sessions from disk")
        except Exception as exc:
            logger.warning(f"Could not load sessions from disk: {exc}")


def _save_sessions():
    try:
        with open(_SESSIONS_FILE, "w") as f:
            json.dump(_session_store, f)
    except Exception as exc:
        logger.warning(f"Could not save sessions to disk: {exc}")


import logging
logger = logging.getLogger(__name__)
_init_sessions()


@dataclass
class Session:
    token: str
    user_id: str
    username: str
    is_admin: bool
    expires_at: float

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


def _create_session(user: User) -> Session:
    """Create a new session for the user."""
    token = secrets.token_urlsafe(32)
    session = Session(
        token=token,
        user_id=user.id,
        username=user.username,
        is_admin=user.is_admin,
        expires_at=time.time() + SESSION_EXPIRE_SECONDS,
    )
    with _session_lock:
        _session_store[token] = {
            "user_id": user.id,
            "username": user.username,
            "is_admin": user.is_admin,
            "expires_at": session.expires_at,
        }
        _save_sessions()
    return session


def _get_session(token: str) -> Optional[Session]:
    """Retrieve a session by token, or None if expired/missing."""
    with _session_lock:
        data = _session_store.get(token)
    if data is None:
        return None
    session = Session(
        token=token,
        user_id=data["user_id"],
        username=data["username"],
        is_admin=data["is_admin"],
        expires_at=data["expires_at"],
    )
    if session.is_expired():
        _delete_session(token)
        return None
    return session


def _delete_session(token: str) -> None:
    with _session_lock:
        _session_store.pop(token, None)
        _save_sessions()


# ── Login helpers ──────────────────────────────────────────────────────────────

def login(username: str, password: str) -> Session:
    """Authenticate user and return a session. Raises HTTPException on failure."""
    from web_fastapi.user_store import verify_password

    user = get_user_by_username(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    if not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return _create_session(user)


def logout(token: str) -> None:
    _delete_session(token)


# ── FastAPI dependency ────────────────────────────────────────────────────────

SESSION_COOKIE = "session_token"


def get_current_user(request: Request) -> Session:
    """Dependency: extract and validate session from cookie."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    session = _get_session(token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid",
        )
    return session


def get_current_admin(request: Request) -> Session:
    """Dependency: require admin session."""
    session = get_current_user(request)
    if not session.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return session
