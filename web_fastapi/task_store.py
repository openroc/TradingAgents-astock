"""Per-user task store — one tasks_{user_id}.json per user, no cross-process lock needed."""

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


def _strip_lc_message(obj: Any) -> Any:
    """Recursively convert LangChain message objects to their text content for JSON serialisation."""
    try:
        from langchain_core.messages import BaseMessage
        from langchain_core.messages.base import BaseMessage as LCMBase
    except ImportError:
        return obj
    if isinstance(obj, (LCMBase, BaseMessage)):
        return obj.content or str(obj)
    if isinstance(obj, dict):
        return {k: _strip_lc_message(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_lc_message(x) for x in obj]
    return obj

# ── Data directory ──────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "userdata"
_OLD_TASKS_FILE = _DATA_DIR / "tasks.json"


def _init_data_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Per-user store cache ───────────────────────────────────────────────────────

_store_cache: dict[str, TaskStore] = {}
_cache_lock = threading.Lock()


def _tasks_file(user_id: str) -> Path:
    # user_id already contains the full prefix, e.g. "u_0a9abdb0"
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

    This function must NOT modify _store_cache during iteration, so it first
    scans files read-only to find the owning user, then loads that one store.
    """
    # 1. Check in-memory cache (no cache modification here)
    with _cache_lock:
        for store in list(_store_cache.values()):
            with store._lock:
                if task_id in store._tasks:
                    return store, store._tasks[task_id]

    # 2. Scan per-user files read-only to find the task's user_id
    uid_found = None
    if _DATA_DIR.exists():
        for f in _DATA_DIR.iterdir():
            if f.name.startswith("tasks_") and f.name.endswith(".json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    for d in data.get("tasks", []):
                        if d.get("task_id") == task_id:
                            uid_found = d.get("user_id")
                            break
                except Exception:
                    pass
            if uid_found:
                break

    # 3. Fall back to old shared tasks.json
    if uid_found is None and _OLD_TASKS_FILE.exists():
        try:
            data = json.loads(_OLD_TASKS_FILE.read_text(encoding="utf-8"))
            for d in data.get("tasks", []):
                if d.get("task_id") == task_id:
                    uid_found = d.get("user_id") or "unknown"
                    break
        except Exception:
            pass

    # Now load the owning store (this may modify _store_cache safely)
    uid = uid_found or "unknown"
    store = get_store(uid)
    with store._lock:
        return store, store._tasks.get(task_id)


# ── Task model ──────────────────────────────────────────────────────────────────

@dataclass
class AnalysisTask:
    """Represents a single analysis task."""

    task_id: str
    user_id: str
    ticker: str
    trade_date: str
    status: str = "pending"  # "pending" | "running" | "completed" | "error" | "abort"
    queued_at: Optional[float] = None  # timestamp when queued for later start
    error: Optional[str] = None
    tracker: ProgressTracker = field(default_factory=ProgressTracker)
    created_at: float = field(default_factory=time.time)
    config: dict = field(default_factory=dict)
    report_summary: Optional[dict] = None
    total_elapsed: Optional[float] = None  # fixed final duration for completed tasks
    final_state: Optional[dict] = None


# ── TaskStore ───────────────────────────────────────────────────────────────────

class TaskStore:
    """Thread-safe per-user task store with JSON persistence."""

    def __init__(self, user_id: str):
        _init_data_dir()
        self._user_id = user_id
        self._tasks_file = _tasks_file(user_id)
        self._lock = threading.Lock()
        self._tasks: dict[str, AnalysisTask] = {}
        self._dirty = False
        self._migrated = False  # prevents duplicate migration
        # Queue: FIFO list of task_ids waiting for a free concurrency slot
        self._queue: list[str] = []
        self._dispatch_event = threading.Event()
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatch_thread.start()
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _task_to_dict(self, task: AnalysisTask) -> dict:
        return {
            "task_id": task.task_id,
            "user_id": task.user_id,
            "ticker": task.ticker,
            "trade_date": task.trade_date,
            "status": task.status,
            "queued_at": task.queued_at,
            "error": task.error,
            "created_at": task.created_at,
            "signal": task.tracker.signal,
            "elapsed": task.tracker.elapsed,
            "total_elapsed": task.total_elapsed,
            "start_time": task.tracker.start_time,
            "llm_calls": task.tracker.llm_calls,
            "tool_calls": task.tracker.tool_calls,
            "tokens_in": task.tracker.tokens_in,
            "tokens_out": task.tracker.tokens_out,
            "completed_stages": task.tracker.completed_stages,
            "final_state": _strip_lc_message(task.final_state) if hasattr(task, "final_state") and task.final_state else {},
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
        # Restore saved elapsed/start_time so list UI shows correct duration
        # elapsed is a read-only property (computed from start_time), so we
        # derive the correct start_time to match the saved elapsed value
        saved_elapsed = d.get("elapsed")
        if saved_elapsed is not None:
            tracker.start_time = time.time() - saved_elapsed
        # Restore total_elapsed so completed tasks show the fixed final duration
        total_elapsed = d.get("total_elapsed")
        return AnalysisTask(
            task_id=d["task_id"],
            user_id=d["user_id"],
            ticker=d["ticker"],
            trade_date=d["trade_date"],
            status=d["status"],
            queued_at=d.get("queued_at"),
            error=d.get("error"),
            created_at=d["created_at"],
            tracker=tracker,
            report_summary=d.get("report_summary"),
            total_elapsed=d.get("total_elapsed"),
            final_state=d.get("final_state"),
            config=d.get("config", {}),
        )

    def _load(self) -> None:
        """Load tasks for this user, migrating from old tasks.json if needed."""
        # Migration: if per-user file doesn't exist, extract from old tasks.json
        if not self._tasks_file.exists():
            self._migrate_from_old_tasks()
            self._migrated = True
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
            # Recover queue: tasks with queued_at that were not started (server restart during queue)
            recovered = 0
            with self._lock:
                for t in self._tasks.values():
                    if t.queued_at is not None and t.status not in ("running", "completed", "error", "abort"):
                        self._queue.append(t.task_id)
                        recovered += 1
            if recovered:
                logger.info("Recovered %d queued tasks for user %s", recovered, self._user_id)
                self.dispatch_queued()
        except Exception as exc:
            logger.warning("Failed to load tasks for %s: %s", self._user_id, exc)

    def _migrate_from_old_tasks(self) -> None:
        """Extract this user's tasks from the old shared tasks.json and remove them from it."""
        if not _OLD_TASKS_FILE.exists():
            logger.info("No old tasks.json found, starting fresh for user %s", self._user_id)
            self._migrated = True
            return

        try:
            data = json.loads(_OLD_TASKS_FILE.read_text(encoding="utf-8"))
            remaining = []
            migrated = []
            for d in data.get("tasks", []):
                if d.get("user_id") == self._user_id:
                    t = self._dict_to_task(d)
                    self._tasks[t.task_id] = t
                    migrated.append(t.task_id)
                else:
                    remaining.append(d)

            if migrated:
                self._dirty = True
                self._save()
                try:
                    _OLD_TASKS_FILE.write_text(
                        json.dumps({"version": 1, "tasks": remaining}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    logger.info(
                        "Migrated %d tasks for user %s and cleaned old tasks.json",
                        len(migrated),
                        self._user_id,
                    )
                except Exception as exc:
                    logger.warning("Failed to clean old tasks.json after migration: %s", exc)
            else:
                logger.info("No tasks to migrate for user %s", self._user_id)
        except Exception as exc:
            logger.warning("Migration failed for user %s: %s", self._user_id, exc)
        finally:
            self._migrated = True

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

    def sync_tracker(self, task_id: str, tracker) -> None:
        """Persist tracker state for a running task (called periodically by runner)."""
        with self._lock:
            if task_id not in self._tasks:
                return
            t = self._tasks[task_id]
            t.tracker.current_stage = tracker.current_stage
            t.tracker.completed_stages = list(tracker.completed_stages)
            t.tracker.stage_reports = {k: _strip_lc_message(v) for k, v in tracker.stage_reports.items()}
            t.tracker.llm_calls = tracker.llm_calls
            t.tracker.tool_calls = tracker.tool_calls
            t.tracker.tokens_in = tracker.tokens_in
            t.tracker.tokens_out = tracker.tokens_out
            t.tracker.start_time = tracker.start_time
            self._dirty = True
        if self._dirty:
            self._save()

    # ── Task queue ─────────────────────────────────────────────────────────────

    def count_running(self) -> int:
        """Count running tasks for this user."""
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == "running")

    def count_queued(self) -> int:
        """Count queued tasks for this user."""
        with self._lock:
            return len(self._queue)

    def queue_push(self, task_id: str) -> None:
        """Add a task to the queue (caller holds _lock)."""
        self._queue.append(task_id)
        self._dirty = True

    def queue_pop(self) -> Optional[str]:
        """Pop the oldest task from the queue (FIFO). Returns None if empty. Caller holds _lock."""
        if self._queue:
            tid = self._queue.pop(0)
            self._dirty = True
            return tid
        return None

    def _dispatch_loop(self) -> None:
        """Background thread: waits for dispatch signal and starts queued tasks as slots free up."""
        while True:
            self._dispatch_event.wait(timeout=1.0)
            self._dispatch_event.clear()
            # Dequeue and start tasks until queue is empty or no free slots
            while True:
                max_conc = self._get_max_concurrency()
                with self._lock:
                    running = sum(1 for t in self._tasks.values() if t.status == "running")
                    if running >= max_conc or not self._queue:
                        break
                    tid = self.queue_pop()
                    if tid is None:
                        break
                    task = self._tasks.get(tid)
                    if task is None:
                        continue
                    task.status = "running"
                    task.queued_at = None
                    self._dirty = True
                    task_config = task.config
                    task_ticker = task.ticker
                    task_trade_date = task.trade_date
                # Spawn analysis thread outside the lock
                try:
                    from web_fastapi.runner import start_analysis
                    start_analysis(tid, task_ticker, task_trade_date, task_config)
                    logger.info("[%s] Queued task dispatched for user %s", tid, self._user_id)
                except Exception as exc:
                    logger.exception("[%s] Failed to dispatch queued task: %s", tid, exc)
                    with self._lock:
                        self._tasks[tid].status = "error"
                        self._tasks[tid].error = f"Dispatch failed: {exc}"
                        self._dirty = True
                self._save()

    def _get_max_concurrency(self) -> int:
        """Load this user's max_concurrency from user_store (outside the lock to avoid deadlock)."""
        try:
            from web_fastapi.user_store import get_user_by_id
            user = get_user_by_id(self._user_id)
            return user.max_concurrency if user else 2
        except Exception:
            return 2

    def dispatch_queued(self) -> None:
        """Signal the dispatch thread to start any queued tasks. Thread-safe."""
        self._dispatch_event.set()

    # ── CRUD ────────────────────────────────────────────────────────────────────

    def create_task(
        self,
        user_id: str,
        ticker: str,
        trade_date: str,
        config: dict,
        max_queue_size: int = 10,
    ) -> str:
        """Create a new analysis task and start or queue it.

        If a concurrency slot is free, the task starts immediately.
        Otherwise it is queued (FIFO) up to max_queue_size tasks.
        Raises RuntimeError if both slots and queue are full.
        """
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
            running = sum(1 for t in self._tasks.values() if t.status == "running")
            max_conc = self._get_max_concurrency()
            if running < max_conc:
                # Start immediately
                task.status = "running"
                task_config = config
                task_ticker = ticker
                task_trade_date = trade_date
            elif len(self._queue) < max_queue_size:
                # Queue for later
                task.queued_at = time.time()
                self._queue.append(task_id)
                logger.info("[%s] Task queued (slot full) for user %s", task_id, user_id)
                self._save()
                return task_id
            else:
                # Queue full — caller should reject with 429
                raise RuntimeError(f"Queue full (max {max_queue_size})")
        self._save()
        if task.status == "running":
            from web_fastapi.runner import start_analysis
            start_analysis(task_id, task_ticker, task_trade_date, task_config)
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
        # When a running task finishes, free up a slot for queued tasks
        if status in ("completed", "error", "abort"):
            self.dispatch_queued()

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
                task.total_elapsed = tracker.elapsed
                task.report_summary = report_summary
                task.final_state = final_state
                self._dirty = True
        if self._dirty:
            self._save()
        self.dispatch_queued()

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
                "elapsed": t.total_elapsed if t.total_elapsed is not None else t.tracker.elapsed,
            })
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }

    def delete_task(self, task_id: str) -> bool:
        """Delete a task. Returns True if the task was deleted."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            was_running = self._tasks[task_id].status == "running"
            # Remove from queue if queued
            if task_id in self._queue:
                self._queue.remove(task_id)
            del self._tasks[task_id]
            self._dirty = True
        self._save()
        if was_running:
            self.dispatch_queued()
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
        self.dispatch_queued()
        return True
