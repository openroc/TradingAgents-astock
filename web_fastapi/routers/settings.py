"""Settings and billing endpoints (auth required)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pathlib import Path

from web_fastapi.auth import Session, get_current_user
from web_fastapi.user_store import get_user_by_id
from web_fastapi.billing_store import get_billing_store

_templates = Jinja2Templates(
    directory=Path(__file__).parent.parent / "templates"
)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", include_in_schema=False)
def settings_page(request: Request):
    """Serve the settings / billing page (requires auth)."""
    get_current_user(request)
    return _templates.TemplateResponse(request, "settings.html", {"request": request})


@router.get("/billing")
def billing_data(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_current_user),
):
    """Return cumulative cost and paginated billing records as JSON."""
    user = get_user_by_id(session.user_id)
    if user is None:
        return {"cumulative_cost": 0.0, "input_price_per_million": 2.10, "output_price_per_million": 8.40, "total": 0, "records": []}
    billing = get_billing_store(session.user_id)
    result = billing.get_records(page=page, page_size=page_size)
    result["input_price_per_million"] = user.input_price_per_million
    result["output_price_per_million"] = user.output_price_per_million
    return result
