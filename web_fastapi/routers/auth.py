"""Authentication endpoints — login, logout, current user."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel, Field
from pathlib import Path

from web_fastapi.auth import (
    login as _do_login,
    logout as _do_logout,
    get_current_user,
    SESSION_COOKIE,
    SESSION_EXPIRE_SECONDS,
)
from web_fastapi.user_store import verify_password, update_user, get_user_by_id


_templates = Jinja2Templates(
    directory=Path(__file__).parent.parent / "templates"
)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=4)


@router.get("/login", include_in_schema=False)
def login_page(request: Request):
    """Serve the login page."""
    return _templates.TemplateResponse(request, "login.html")


@router.post("/login")
def login(req: LoginRequest):
    """Authenticate and set session cookie."""
    session = _do_login(req.username, req.password)
    response = Response(
        content=f'{{"ok": true, "token": "{session.token}"}}',
        media_type="application/json",
    )
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session.token,
        max_age=SESSION_EXPIRE_SECONDS,
        httponly=True,
        samesite="lax",
    )
    return response


@router.post("/logout")
def logout(request: Request):
    """Clear session cookie."""
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        _do_logout(token)
    response = Response(content='{"ok": true}', media_type="application/json")
    response.delete_cookie(SESSION_COOKIE)
    return response


@router.get("/me")
def me(request: Request):
    """Return current user info (requires auth)."""
    session = get_current_user(request)
    return {
        "user_id": session.user_id,
        "username": session.username,
        "is_admin": session.is_admin,
    }


@router.get("/change-password", include_in_schema=False)
def change_password_page(request: Request):
    """Serve the change-password page (requires auth)."""
    get_current_user(request)
    return _templates.TemplateResponse(request, "change_password.html")


@router.put("/password")
def change_password(req: ChangePasswordRequest, request: Request):
    """Change current user's password. Verifies current password first."""
    session = get_current_user(request)
    user = get_user_by_id(session.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    update_user(session.user_id, password=req.new_password)
    return {"ok": True}
