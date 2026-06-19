"""Per-user task store — one tasks_{user_id}.json per user, no cross-process lock needed."""

from __future__ import annotations

import json
import logging
import threading
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from web.progress import ProgressTracker

logger = logging.getLogger(__name__)

# ── Data directory ──────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"
_OLD_TASKS_FILE = _DATA_DIR / "tasks.json"


def _init_data_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Per-user store cache ───────────────────────────────────────────────────────

_store_cache: dict[str, TaskStore] = {}
_cache_lock = threading.Lock()


def _tasks_file(user_id: str) -> Path:
    return _DATA_DIR / f"tasks_{user_id}.json"


def get_store(user_id: str) -> TaskStore:
    """Get or create the TaskStore for a given user (cached per process)."""
    with _cache_lock:
        if user_id not in _store_cache:
            _store_cache[user_id] = TaskStore(user_id)
        return _store_cache[user_id]


def get_store_for_task(task_id: str) -> tuple[TaskStore, Optional[AnalysisTask]]:
    """Find the store containing a task by task_id.

    Searches per-user stores (cached + on disk) and falls back to the old
    shared tasks.json for tasks not yet migrated. Used by runner.py which only
    knows task_id, not user_id.
    """
    # 1. Check in-memory cache of per-user stores
    with _cache_lock:
        for uid, store in _store_cache.items():
            with store._lock:
                if task_id in store._tasks:
                    return store, store._tasks[task_id]

    # 2. Check per-user files not yet loaded into cache
    if _DATA_DIR.exists():
        for f in _DATA_DIR.iterdir():
            if f.name.startswith("tasks_") and f.name.endswith(".json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    for d in data.get("tasks", []):
                        if d.get("task_id") == task_id:
                            uid = f.stem  # "tasks_u_xxx" → "u_xxx"
                            store = get_store(uid)
                            with store._lock:
                                if task_id in store._tasks:
                                    return store, store._tasks[task_id]
                except Exception:
                    pass

    # 3. Fall back to old shared tasks.json (pre-migration)
    if _OLD_TASKS_FILE.exists():
        try:
            data = json.loads(_OLD_TASKS_FILE.read_text(encoding="utf-8"))
            for d in data.get("tasks", []):
                if d.get("task_id") == task_id:
                    uid = d.get("user_id", "unknown")
                    store = get_store(uid)
                    with store._lock:
                        if task_id in store._tasks:
                            return store, store._tasks[task_id]
        except Exception:
            pass

    return get_store("unknown"), None


# ── Task model ──────────────────────────────────────────────────────────────────

@dataclass
class AnalysisTask:
    """Represents a single analysis task."""

    task_id: str
    user_id: str
    ticker: str
    trade_date: str
    status: str = "pending"  # "pending" | "running" | "completed" | "error" | "abort"
    error: Optional[str] = None
    tracker: ProgressTracker = field(default_factory=ProgressTracker)
    created_at: float = field(default_factory=time.time)
    config: dict = field(default_factory=dict)
    report_summary: Optional[dict] = None
    final_state: Optional[dict] = None


# ── TaskStore ───────────────────────────────────────────────────────────────────

class TaskStore:
    """Thread-safe per-user task store with JSON persistence."""

    _tasks: dict[str, AnalysisTask] = {}
    _lock = threading.Lock()
    _dirty = False  # tracks whether we need to persist

    def __init__(self, user_id: str):
        _init_data_dir()
        self._user_id = user_id
        self._tasks_file = _tasks_file(user_id)
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _task_to_dict(self, task: AnalysisTask) -> dict:
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
        """Load tasks for this user, migrating from old tasks.json if needed."""
        # Migration: if per-user file doesn't exist, extract from old tasks.json
        if not self._tasks_file.exists():
            self._migrate_from_old_tasks()
            return

        try:
            data = json.loads(self._tasks_file.read_text(encoding="utf-8"))
            with self._lock:
                for d in data.get("tasks", []):
                    t = self._dict_to_task(d)
                    self._tasks[t.task_id] = t
            logger.info(
                "Loaded %d tasks for user %s from %s",
                len(self._tasks),
                self._user_id,
                self._tasks_file.name,
            )
        except Exception as exc:
            logger.warning("Failed to load tasks for %s: %s", self._user_id, exc)

    def _migrate_from_old_tasks(self) -> None:
        """Extract this user's tasks from the old shared tasks.json."""
        if not _OLD_TASKS_FILE.exists():
            logger.info("No old tasks.json found, starting fresh for user %s", self._user_id)
            return

        try:
            data = json.loads(_OLD_TASKS_FILE.read_text(encoding="utf-8"))
            migrated = []
            for d in data.get("tasks", []):
                if d.get("user_id") == self._user_id:
                    t = self._dict_to_task(d)
                    self._tasks[t.task_id] = t
                    migrated.append(t.task_id)

            if migrated:
                self._dirty = True
                self._save()
                logger.info(
                    "Migrated %d tasks for user %s to %s",
                    len(migrated),
                    self._user_id,
                    self._tasks_file.name,
                )
            else:
                logger.info("No tasks to migrate for user %s", self._user_id)
        except Exception as exc:
            logger.warning("Migration failed for user %s: %s", self._user_id, exc)

    def _save(self) -> None:
        """Persist tasks to this user's file (caller holds _lock)."""
        try:
            tasks_list = [self._task_to_dict(t) for t in self._tasks.values()]
        except Exception as exc:
            logger.warning("Failed to serialise tasks for %s: %s", self._user_id, exc)
            return
        data = {"version": 1, "tasks": tasks_list}
        try:
            self._tasks_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception as exc:
            logger.warning("Failed to persist tasks for %s: %s", self._user_id, exc)

    # ── CRUD ────────────────────────────────────────────────────────────────────

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
            self._dirty = True
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
                self._dirty = True
        if self._dirty:
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
                self._dirty = True
        if self._dirty:
            self._save()

    def list_tasks(self, page: int = 1, page_size: int = 20) -> dict:
        """List tasks for this user, with pagination."""
        with self._lock:
            tasks = sorted(
                self._tasks.values(),
                key=lambda t: t.created_at,
                reverse=True,
            )
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
                "error": t.error,
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
            if task_id not in self._tasks:
                return False
            del self._tasks[task_id]
            self._dirty = True
        self._save()
        return True

    def abort_task(self, task_id: str, reason: str = "任务被手动终止") -> bool:
        """Mark a running task as aborted."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            self._tasks[task_id].status = "abort"
            self._tasks[task_id].error = reason
            self._dirty = True
        self._save()
        return True

    def count_running(self) -> int:
        """Count running tasks for this user."""
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == "running")
