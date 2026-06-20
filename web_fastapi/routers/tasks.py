"""Task management endpoints — create and poll analysis tasks (auth required)."""

from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request as StarletteRequest
from pydantic import BaseModel, Field
from pathlib import Path

from web_fastapi.task_store import get_store
from web_fastapi.config import build_config
from web_fastapi import auth
from web_fastapi.auth import Session
from web_fastapi.user_store import get_user_by_id


_templates = Jinja2Templates(
    directory=Path(__file__).parent.parent / "templates"
)

router = APIRouter(prefix="/tasks", tags=["tasks"])


# ── Page routes (no auth on template rendering) ──────────────────────────────────

@router.get("", include_in_schema=False)
def tasks_page(request: Request):
    return _templates.TemplateResponse(request, "tasks_list.html", {"request": request})


# ── API routes (auth required) ────────────────────────────────────────────────

class CreateTaskRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20, description="Stock ticker")
    trade_date: Optional[str] = Field(
        default=None,
        description="Trade date YYYY-MM-DD (defaults to today)",
    )
    llm_provider: Optional[str] = None
    deep_think_llm: Optional[str] = None
    quick_think_llm: Optional[str] = None
    backend_url: Optional[str] = None
    max_debate_rounds: int = Field(default=1, ge=1, le=5)
    max_risk_discuss_rounds: int = Field(default=1, ge=1, le=5)
    output_language: str = Field(default="Chinese")
    data_vendors: Optional[dict] = None


class CreateTaskResponse(BaseModel):
    task_id: str
    ticker: str
    trade_date: str
    status: str
    poll_url: str


class TaskStatusResponse(BaseModel):
    task_id: str
    ticker: str
    trade_date: str
    status: str
    error: Optional[str] = None
    current_stage: str
    completed_stages: list[str]
    stage_reports: dict[str, str]
    signal: str
    llm_calls: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    elapsed_seconds: float
    cost: Optional[float] = None
    report_url: Optional[str] = None


@router.post("", response_model=CreateTaskResponse, status_code=201)
def create_task(
    req: CreateTaskRequest,
    session: Session = Depends(auth.get_current_user),
):
    """Submit a new analysis task. Task runs in background."""
    user = get_user_by_id(session.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    store = get_store(session.user_id)

    trade_date = req.trade_date or str(date.today())

    # Use user's defaults if not specified
    provider = req.llm_provider or user.default_provider
    deep_model = req.deep_think_llm or user.default_model
    quick_model = req.quick_think_llm or user.default_model

    config = build_config(
        llm_provider=provider,
        deep_think_llm=deep_model,
        quick_think_llm=quick_model,
        backend_url=req.backend_url,
        max_debate_rounds=req.max_debate_rounds,
        max_risk_discuss_rounds=req.max_risk_discuss_rounds,
        output_language=req.output_language,
        data_vendors=req.data_vendors,
    )

    try:
        task_id = store.create_task(session.user_id, req.ticker, trade_date, config)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        )

    return CreateTaskResponse(
        task_id=task_id,
        ticker=req.ticker,
        trade_date=trade_date,
        status="pending",
        poll_url=f"/tasks/{task_id}",
    )


@router.get("/list", response_model=dict)
def list_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(auth.get_current_user),
) -> dict:
    """List tasks for the current user."""
    return get_store(session.user_id).list_tasks(page=page, page_size=page_size)


@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_task_status(
    task_id: str,
    session: Session = Depends(auth.get_current_user),
) -> TaskStatusResponse:
    """Poll the current status of an analysis task."""
    task = get_store(session.user_id).get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id != session.user_id and not session.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")

    tracker = task.tracker
    report_url = f"/report/{task_id}" if task.status == "completed" else None

    return TaskStatusResponse(
        task_id=task_id,
        ticker=task.ticker,
        trade_date=task.trade_date,
        status=task.status,
        error=task.error,
        current_stage=tracker.current_stage,
        completed_stages=tracker.completed_stages,
        stage_reports=tracker.stage_reports,
        signal=tracker.signal,
        llm_calls=tracker.llm_calls,
        tool_calls=tracker.tool_calls,
        tokens_in=tracker.tokens_in,
        tokens_out=tracker.tokens_out,
        elapsed_seconds=round(tracker.elapsed, 1),
        cost=task.cost,
        report_url=report_url,
    )


@router.post("/{task_id}/delete")
def delete_task_post(
    request: Request,
    task_id: str,
    session: Session = Depends(auth.get_current_user),
):
    """Delete a task. Form submit redirects to /tasks; AJAX returns JSON."""
    store = get_store(session.user_id)
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id != session.user_id and not session.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    store.delete_task(task_id)

    ct = request.headers.get("content-type", "")
    if "application/json" not in ct:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/tasks", status_code=302)
    return {"ok": True}


@router.post("/{task_id}/rerun")
def rerun_task(
    task_id: str,
    session: Session = Depends(auth.get_current_user),
):
    """Delete the old task and start a new one with the same parameters."""
    store = get_store(session.user_id)
    old_task = store.get_task(task_id)
    if old_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if old_task.user_id != session.user_id and not session.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")

    user = get_user_by_id(session.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    # Save config before deleting
    if old_task.config:
        config = old_task.config
    else:
        config = build_config(
            llm_provider=user.default_provider,
            deep_think_llm=user.default_model,
            quick_think_llm=user.default_model,
        )

    # Delete old task first (in case its status is stuck on "running")
    store.delete_task(task_id)

    # create_task now handles concurrency/queue automatically
    try:
        new_id = store.create_task(
            session.user_id,
            old_task.ticker,
            old_task.trade_date,
            config,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        )

    return {"task_id": new_id, "ticker": old_task.ticker, "status": "pending"}


@router.post("/{task_id}/abort")
def abort_task(
    task_id: str,
    session: Session = Depends(auth.get_current_user),
):
    """Mark a stuck running task as aborted (e.g. process was killed)."""
    store = get_store(session.user_id)
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id != session.user_id and not session.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    store.abort_task(task_id, "任务被手动终止")
    return {"ok": True}
