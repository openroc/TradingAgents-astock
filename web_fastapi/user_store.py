"""User store — loads/saves users from JSON, with bcrypt password hashing."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_USERS_FILE = _DATA_DIR / "users.json"

_DEFAULT_ADMIN_PASSWORD = "admin"  # change after first login


@dataclass
class User:
    id: str
    username: str
    password_hash: str
    is_admin: bool = False
    max_concurrency: int = 2
    default_model: str = "MiniMax-M2.7"
    default_provider: str = "oneapi"


def _init_data_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Password hashing (bcrypt via passlib or fallback sha256) ───────────────────

import hashlib
import io
import sys

_has_bcrypt = False
try:
    # Suppress passlib's "trapped error reading bcrypt version" traceback on stderr
    _old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        from passlib.context import CryptContext

        _pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        # Test round-trip to catch bcrypt 5.x / passlib incompatibility
        _test_hash = _pwd_ctx.hash("__passlib_test__")
        assert _pwd_ctx.verify("__passlib_test__", _test_hash)
        _has_bcrypt = True
    finally:
        sys.stderr = _old_stderr
except Exception:
    # bcrypt unavailable or broken (e.g. bcrypt 5.x + passlib incompatibility) — use sha256
    sys.stderr = _old_stderr if "_old_stderr" in dir() else sys.stderr
    _pwd_ctx = None
    logger.warning("bcrypt unavailable; using sha256 for password hashing (less secure)")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (preferred) or sha256 (fallback)."""
    if _has_bcrypt:
        return _pwd_ctx.hash(password)
    salt = uuid.uuid4().hex[:16]
    return f"sha256${salt}${hashlib.sha256((salt + password).encode()).hexdigest()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    # sha256 fallback hashes must be verified manually
    if hashed.startswith("sha256$"):
        try:
            _, salt, stored = hashed.split("$", 2)
            return hashlib.sha256((salt + password).encode()).hexdigest() == stored
        except Exception:
            return False
    # bcrypt hashes via passlib
    if _has_bcrypt:
        return _pwd_ctx.verify(password, hashed)
    return False


# ── Store ──────────────────────────────────────────────────────────────────────

_users: dict[str, User] = {}
_lock = __import__("threading").Lock()


def _load() -> None:
    _init_data_dir()
    if not _USERS_FILE.exists():
        _create_default_users()
        return
    try:
        data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
        with _lock:
            _users.clear()
            for d in data.get("users", []):
                u = User(
                    id=d["id"],
                    username=d["username"],
                    password_hash=d["password_hash"],
                    is_admin=d.get("is_admin", False),
                    max_concurrency=d.get("max_concurrency", 2),
                    default_model=d.get("default_model", "MiniMax-M2.7"),
                    default_provider=d.get("default_provider", "oneapi"),
                )
                _users[u.id] = u
        logger.info("Loaded %d users", len(_users))
    except Exception as exc:
        logger.warning("Failed to load users: %s; creating defaults", exc)
        _create_default_users()


def _save() -> None:
    _init_data_dir()
    try:
        with _lock:
            users_list = [
                {
                    "id": u.id,
                    "username": u.username,
                    "password_hash": u.password_hash,
                    "is_admin": u.is_admin,
                    "max_concurrency": u.max_concurrency,
                    "default_model": u.default_model,
                    "default_provider": u.default_provider,
                }
                for u in _users.values()
            ]
        _USERS_FILE.write_text(
            json.dumps({"version": 1, "users": users_list}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to save users: %s", exc)


def _create_default_users() -> None:
    """Create a default admin user. User must change password on first login."""
    admin = User(
        id="u_default_admin",
        username="admin",
        password_hash=hash_password(_DEFAULT_ADMIN_PASSWORD),
        is_admin=True,
        max_concurrency=10,
        default_model="MiniMax-M2.7",
        default_provider="oneapi",
    )
    with _lock:
        _users.clear()
        _users[admin.id] = admin
    _save()
    logger.info("Created default admin user (username: admin, password: admin) — change it immediately!")


# ── Public API ─────────────────────────────────────────────────────────────────

def get_user_by_username(username: str) -> Optional[User]:
    with _lock:
        for u in _users.values():
            if u.username == username:
                return u
    return None


def get_user_by_id(user_id: str) -> Optional[User]:
    with _lock:
        return _users.get(user_id)


def create_user(
    username: str,
    password: str,
    *,
    is_admin: bool = False,
    max_concurrency: int = 2,
    default_model: str = "MiniMax-M2.7",
    default_provider: str = "oneapi",
) -> User:
    """Create a new user. Raises ValueError if username already exists."""
    if get_user_by_username(username) is not None:
        raise ValueError(f"Username '{username}' already exists")
    user = User(
        id=f"u_{uuid.uuid4().hex[:8]}",
        username=username,
        password_hash=hash_password(password),
        is_admin=is_admin,
        max_concurrency=max_concurrency,
        default_model=default_model,
        default_provider=default_provider,
    )
    with _lock:
        _users[user.id] = user
    _save()
    return user


def update_user(
    user_id: str,
    *,
    password: Optional[str] = None,
    is_admin: Optional[bool] = None,
    max_concurrency: Optional[int] = None,
    default_model: Optional[str] = None,
    default_provider: Optional[str] = None,
) -> Optional[User]:
    """Update user fields. Returns None if user not found."""
    with _lock:
        user = _users.get(user_id)
        if user is None:
            return None
        if password is not None:
            user.password_hash = hash_password(password)
        if is_admin is not None:
            user.is_admin = is_admin
        if max_concurrency is not None:
            user.max_concurrency = max_concurrency
        if default_model is not None:
            user.default_model = default_model
        if default_provider is not None:
            user.default_provider = default_provider
    _save()
    return user


def delete_user(user_id: str) -> bool:
    """Delete a user. Returns False if not found."""
    with _lock:
        if user_id not in _users:
            return False
        # Prevent deleting the last admin
        user = _users[user_id]
        if user.is_admin:
            admin_count = sum(1 for u in _users.values() if u.is_admin)
            if admin_count <= 1:
                raise ValueError("Cannot delete the last admin user")
        del _users[user_id]
    _save()
    return True


def list_users() -> list[User]:
    with _lock:
        return list(_users.values())


# ── Init on module load ────────────────────────────────────────────────────────
_load()
