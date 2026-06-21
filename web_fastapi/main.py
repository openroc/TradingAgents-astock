"""FastAPI application entry point — XT系统 Web API."""

from __future__ import annotations

import argparse
from datetime import date
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

_CORS_ORIGINS = [
    "http://localhost:8000",
    "http://localhost:8090",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8090",
]
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from web_fastapi.user_store import get_user_by_id
from web_fastapi.config import DEFAULT_BACKEND_URLS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from web_fastapi.routers import tasks, report, auth as auth_router, admin, settings
from web_fastapi.auth import SESSION_COOKIE, _get_session


def _record_partial_billing_for_running_tasks() -> None:
    """Record billing for all tasks still in 'running' status.

    Called on shutdown (to capture work done before kill) and on startup
    (to handle tasks orphaned by an abrupt previous shutdown).
    """
    try:
        from web_fastapi.task_store import get_store
        from web_fastapi.user_store import list_users
        from web_fastapi.billing_store import get_billing_store
        logger.info("Recording partial billing for interrupted running tasks...")
        count = 0
        for user in list_users():
            store = get_store(user.id)
            for task_id, task in store._tasks.items():
                if task.status == "running" and task.tracker.tokens_in > 0:
                    model_name = task.config.get("deep_think_llm", "unknown")
                    billing = get_billing_store(user.id)
                    user_pricing = None
                    if user.input_price_per_million and user.output_price_per_million:
                        user_pricing = {
                            "input_price_per_million": user.input_price_per_million,
                            "output_price_per_million": user.output_price_per_million,
                        }
                    try:
                        billing.record_task(
                            task_id=task_id,
                            ticker=task.ticker,
                            tokens_in=task.tracker.tokens_in,
                            tokens_out=task.tracker.tokens_out,
                            model_name=model_name,
                            user_pricing=user_pricing,
                        )
                        count += 1
                        logger.info("  Recorded partial billing: %s (%s tokens_in=%d)", task_id, task.ticker, task.tracker.tokens_in)
                    except Exception as exc:
                        logger.warning("  Failed to record partial billing for %s: %s", task_id, exc)
        if count:
            logger.info("Recorded partial billing for %d interrupted tasks", count)
        else:
            logger.info("No interrupted running tasks to bill")
    except Exception as exc:
        logger.warning("Failed to record partial billing on shutdown: %s", exc)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: bill any tasks left in 'running' state by a previous crash/kill
    _record_partial_billing_for_running_tasks()
    logger.info("XT FastAPI server started")
    yield
    # On shutdown: bill any currently running tasks before we die
    _record_partial_billing_for_running_tasks()
    logger.info("XT FastAPI server shutting down")


app = FastAPI(
    title="XT API",
    description="多 Agent A 股投研框架 - 基于 LangGraph 的智能分析服务",
    version="0.2.11",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static & templates ─────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=_PROJECT_ROOT / "assets"), name="static")


# ── Auth redirect for HTML pages ───────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def auth_redirect_handler(request: Request, exc: HTTPException):
    """Redirect unauthenticated HTML requests to login instead of returning JSON 401."""
    if exc.status_code == 401 and "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/auth/login", status_code=302)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

templates = Jinja2Templates(directory=_PROJECT_ROOT / "web_fastapi" / "templates")


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router.router)
app.include_router(admin.router)
app.include_router(tasks.router)
app.include_router(report.router)
app.include_router(settings.router)


# ── Root — check auth, redirect to login if not authenticated ─────────────────
@app.get("/", include_in_schema=False)
def root(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token or not _get_session(token):
        return RedirectResponse(url="/auth/login", status_code=302)
    session = _get_session(token)
    user = get_user_by_id(session.user_id)
    provider = user.default_provider if user else "oneapi"
    model = user.default_model if user else "MiniMax-M2.7"
    backend_url = DEFAULT_BACKEND_URLS.get(provider, "")
    return templates.TemplateResponse(request, "task_form.html", {
        "request": request,
        "today": str(date.today()),
        "user_provider": provider,
        "user_model": model,
        "user_backend_url": backend_url or "",
    })


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": "XT系统"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="XT FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", default=False)
    args = parser.parse_args()

    uvicorn.run(
        "web_fastapi.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
