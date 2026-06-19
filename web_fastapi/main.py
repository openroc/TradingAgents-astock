"""FastAPI application entry point — XT投研辅助系统 Web API."""

from __future__ import annotations

import argparse
from datetime import date
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from web_fastapi.user_store import get_user_by_id
from web_fastapi.config import DEFAULT_BACKEND_URLS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from web_fastapi.routers import tasks, report, auth as auth_router, admin
from web_fastapi.auth import SESSION_COOKIE, _get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("XT投研辅助系统 FastAPI server started")
    yield
    logger.info("XT投研辅助系统 FastAPI server shutting down")


app = FastAPI(
    title="XT投研辅助系统 API",
    description="多 Agent A 股投研框架 - 基于 LangGraph 的智能分析服务",
    version="0.2.11",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static & templates ─────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=_PROJECT_ROOT / "assets"), name="static")

templates = Jinja2Templates(directory=_PROJECT_ROOT / "web_fastapi" / "templates")


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router.router)
app.include_router(admin.router)
app.include_router(tasks.router)
app.include_router(report.router)


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
    return {"status": "ok", "service": "XT投研辅助系统"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="XT投研辅助系统 FastAPI Server")
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
