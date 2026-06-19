"""Thread-safe in-memory task store with disk persistence for multi-user multi-task analysis."""

from __future__ import annotations

import json
import logging
import threading
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from web.progress import ProgressTracker

logger = logging.getLogger(__name__)

# Persist data directory relative to this file
_DATA_DIR = Path(__file__).parent / "data"
_TASKS_FILE = _DATA_DIR / "tasks.json"


def _init_data_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AnalysisTask:
    """Represents a single analysis task."""

    task_id: str
    user_id: str
    ticker: str
    trade_date: str
    status: str = "pending"  # "pending" | "running" | "completed" | "error"
    error: Optional[str] = None
    tracker: ProgressTracker = field(default_factory=ProgressTracker)
    created_at: float = field(default_factory=time.time)
    config: dict = field(default_factory=dict)
    report_summary: Optional[dict] = None
    final_state: Optional[dict] = None


class TaskStore:
    """Thread-safe store with JSON persistence. Loads from disk on init."""

    _tasks: dict[str, AnalysisTask] = {}
    _lock = threading.Lock()

    def __init__(self):
        _init_data_dir()
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _task_to_dict(self, task: AnalysisTask) -> dict:
        """Serialize task to a JSON-safe dict."""
        # Avoid serialising the ProgressTracker directly; store its fields
        return {
            "task_id": task.task_id,
            "user_id": task.user_id,
            "ticker": task.ticker,
            "trade_date": task.trade_date,
            "status": task.status,
            "error": task.error,
            "created_at": task.created_at,
            "signal": task.tracker.signal,
            "elapsed": task.tracker.elapsed,
            "llm_calls": task.tracker.llm_calls,
            "tool_calls": task.tracker.tool_calls,
            "tokens_in": task.tracker.tokens_in,
            "tokens_out": task.tracker.tokens_out,
            "completed_stages": task.tracker.completed_stages,
            "final_state": task.final_state if hasattr(task, "final_state") else {},
            "report_summary": task.report_summary,
            "config": task.config,
        }

    def _dict_to_task(self, d: dict) -> AnalysisTask:
        """Reconstruct a task from a persisted dict."""
        tracker = ProgressTracker(
            ticker=d["ticker"],
            trade_date=d["trade_date"],
        )
        tracker.signal = d.get("signal", "")
        tracker.llm_calls = d.get("llm_calls", 0)
        tracker.tool_calls = d.get("tool_calls", 0)
        tracker.tokens_in = d.get("tokens_in", 0)
        tracker.tokens_out = d.get("tokens_out", 0)
        tracker.completed_stages = d.get("completed_stages", [])
        if d["status"] == "completed":
            tracker.is_complete = True
        return AnalysisTask(
            task_id=d["task_id"],
            user_id=d["user_id"],
            ticker=d["ticker"],
            trade_date=d["trade_date"],
            status=d["status"],
            error=d.get("error"),
            created_at=d["created_at"],
            tracker=tracker,
            report_summary=d.get("report_summary"),
            final_state=d.get("final_state"),
            config=d.get("config", {}),
        )

    def _load(self) -> None:
        """Load persisted tasks from disk."""
        if not _TASKS_FILE.exists():
            return
        try:
            data = json.loads(_TASKS_FILE.read_text(encoding="utf-8"))
            with self._lock:
                for d in data.get("tasks", []):
                    t = self._dict_to_task(d)
                    self._tasks[t.task_id] = t
            logger.info("Loaded %d persisted tasks", len(self._tasks))
        except Exception as exc:
            logger.warning("Failed to load tasks from disk: %s", exc)

    def _save(self) -> None:
        """Persist all tasks to disk."""
        try:
            with self._lock:
                tasks_list = [self._task_to_dict(t) for t in self._tasks.values()]
            data = {"version": 1, "tasks": tasks_list}
            _TASKS_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to persist tasks: %s", exc)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create_task(
        self,
        user_id: str,
        ticker: str,
        trade_date: str,
        config: dict,
    ) -> str:
        """Create a new analysis task and return its ID."""
        task_id = str(uuid.uuid4())[:8]
        task = AnalysisTask(
            task_id=task_id,
            user_id=user_id,
            ticker=ticker,
            trade_date=trade_date,
            status="pending",
            tracker=ProgressTracker(ticker=ticker, trade_date=trade_date),
            config=config,
        )
        with self._lock:
            self._tasks[task_id] = task
        self._save()
        return task_id

    def get_task(self, task_id: str) -> Optional[AnalysisTask]:
        """Retrieve a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update task status and persist."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = status
                self._tasks[task_id].error = error
        self._save()

    def update_completed(
        self,
        task_id: str,
        tracker: ProgressTracker,
        report_summary: Optional[dict] = None,
        final_state: Optional[dict] = None,
    ) -> None:
        """Mark task as completed with full tracker state and persist."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = "completed"
                task.tracker = tracker
                task.report_summary = report_summary
                task.final_state = final_state
        self._save()

    def list_tasks(
        self,
        user_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """List tasks for a specific user, with pagination."""
        with self._lock:
            tasks = sorted(
                self._tasks.values(),
                key=lambda t: t.created_at,
                reverse=True,
            )
        if user_id:
            tasks = [t for t in tasks if t.user_id == user_id]
        total = len(tasks)
        start = (page - 1) * page_size
        end = start + page_size
        items = []
        for t in tasks[start:end]:
            items.append({
                "task_id": t.task_id,
                "user_id": t.user_id,
                "ticker": t.ticker,
                "trade_date": t.trade_date,
                "status": t.status,
                "signal": t.tracker.signal,
                "created_at": t.created_at,
                "elapsed": t.tracker.elapsed,
            })
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._save()
                return True
            return False


# Global singleton — loaded once per process
task_store = TaskStore()
