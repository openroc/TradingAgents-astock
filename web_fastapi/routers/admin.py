"""Admin endpoints — user management (admin only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel, Field
from pathlib import Path

from web_fastapi.auth import Session, get_current_admin
from web_fastapi import auth
from web_fastapi.model_pricing import list_models, update_model_pricing, get_model_pricing
from web_fastapi.user_store import (
    create_user,
    update_user,
    delete_user,
    list_users,
    get_user_by_id,
)


_templates = Jinja2Templates(
    directory=Path(__file__).parent.parent / "templates"
)

router = APIRouter(prefix="/admin", tags=["admin"])


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=4)
    is_admin: bool = False
    max_concurrency: int = Field(default=2, ge=1, le=20)
    default_model: str = "MiniMax-M2.7"
    default_provider: str = "oneapi"
    input_price_per_million: float = Field(default=2.10, ge=0)
    output_price_per_million: float = Field(default=8.40, ge=0)


class UpdateUserRequest(BaseModel):
    password: Optional[str] = None
    is_admin: Optional[bool] = None
    max_concurrency: Optional[int] = Field(default=None, ge=1, le=20)
    default_model: Optional[str] = None
    default_provider: Optional[str] = None
    input_price_per_million: Optional[float] = Field(default=None, ge=0)
    output_price_per_million: Optional[float] = Field(default=None, ge=0)


def _user_to_dict(u) -> dict:
    return {
        "id": u.id,
        "username": u.username,
        "is_admin": u.is_admin,
        "max_concurrency": u.max_concurrency,
        "default_model": u.default_model,
        "default_provider": u.default_provider,
        "input_price_per_million": u.input_price_per_million,
        "output_price_per_million": u.output_price_per_million,
    }


@router.get("/users")
def admin_list_users(session: Session = Depends(auth.get_current_admin)):
    """List all users (admin only)."""
    return [_user_to_dict(u) for u in list_users()]


@router.post("/users")
def admin_create_user(
    req: CreateUserRequest,
    session: Session = Depends(auth.get_current_admin),
):
    """Create a new user (admin only)."""
    try:
        user = create_user(
            username=req.username,
            password=req.password,
            is_admin=req.is_admin,
            max_concurrency=req.max_concurrency,
            default_model=req.default_model,
            default_provider=req.default_provider,
            input_price_per_million=req.input_price_per_million,
            output_price_per_million=req.output_price_per_million,
        )
        return _user_to_dict(user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/users/{user_id}")
def admin_update_user(
    user_id: str,
    req: UpdateUserRequest,
    session: Session = Depends(auth.get_current_admin),
):
    """Update a user (admin only)."""
    user = update_user(
        user_id,
        password=req.password,
        is_admin=req.is_admin,
        max_concurrency=req.max_concurrency,
        default_model=req.default_model,
        default_provider=req.default_provider,
        input_price_per_million=req.input_price_per_million,
        output_price_per_million=req.output_price_per_million,
    )
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_to_dict(user)


@router.delete("/users/{user_id}", status_code=204)
def admin_delete_user(
    user_id: str,
    session: Session = Depends(auth.get_current_admin),
):
    """Delete a user (admin only)."""
    try:
        if not delete_user(user_id):
            raise HTTPException(status_code=404, detail="User not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Model pricing ───────────────────────────────────────────────────────────────

class UpdateModelPricingRequest(BaseModel):
    input_price_per_million: Optional[float] = Field(default=None, ge=0)
    output_price_per_million: Optional[float] = Field(default=None, ge=0)
    weight: Optional[float] = Field(default=None, ge=0)


@router.get("/models")
def admin_list_models(session: Session = Depends(auth.get_current_admin)):
    """List all model pricing configs (admin only)."""
    return list_models()


@router.get("/models/{model_name}")
def admin_get_model(
    model_name: str,
    session: Session = Depends(auth.get_current_admin),
):
    """Get pricing for a specific model (admin only)."""
    return get_model_pricing(model_name)


@router.put("/models/{model_name}")
def admin_update_model_pricing(
    model_name: str,
    req: UpdateModelPricingRequest,
    session: Session = Depends(auth.get_current_admin),
):
    """Update pricing for a specific model (admin only)."""
    updates = {}
    if req.input_price_per_million is not None:
        updates["input_price_per_million"] = req.input_price_per_million
    if req.output_price_per_million is not None:
        updates["output_price_per_million"] = req.output_price_per_million
    if req.weight is not None:
        updates["weight"] = req.weight
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        updated = update_model_pricing(model_name, updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in models.json")
    return {"ok": True, "model": model_name, "updates": updates}
