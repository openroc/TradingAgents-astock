"""LangGraph checkpoint support for resumable analysis runs.

Per-ticker SQLite databases so concurrent tickers don't contend.
"""

from __future__ import annotations

import hashlib
import io
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from langgraph.checkpoint.sqlite import SqliteSaver

from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)


# ── Execution log ──────────────────────────────────────────────────────────────


class GraphExecutionLogger:
    """Thread-safe per-task file logger for errors and warnings during graph runs.

    One log file per ticker+date, written incrementally (auto-flush) so it can
    be inspected mid-run.  Log file lives next to the checkpoint DB::

        ~/.tradingagents/cache/checkpoints/
            SPY.db          ← SqliteSaver checkpoint DB
            SPY_2025-01-15.log  ← execution log
    """

    _lock = threading.Lock()

    def __init__(self, path: Path):
        self._path = path
        self._file: Optional[io.TextIOWrapper] = None

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self._path, "a", encoding="utf-8", buffering=1)

    def _write(self, level: str, msg: str):
        with self._lock:
            self._ensure_open()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._file.write(f"[{ts}] [{level}] {msg}\n")
            self._file.flush()

    def error(self, msg: str):
        self._write("ERROR", msg)

    def warning(self, msg: str):
        self._write("WARNING", msg)

    def info(self, msg: str):
        self._write("INFO", msg)

    def close(self):
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _log_path(data_dir: str | Path, ticker: str, date: str) -> Path:
    """Return the execution log path for a ticker+date."""
    safe = safe_ticker_component(ticker).upper()
    cp_dir = Path(data_dir) / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    return cp_dir / f"{safe}_{date}.log"


def get_execution_logger(
    data_dir: str | Path, ticker: str, date: str
) -> GraphExecutionLogger:
    """Create (or append to) an execution log for the given ticker+date."""
    return GraphExecutionLogger(_log_path(data_dir, ticker, date))


# ── Checkpoint DB ───────────────────────────────────────────────────────────────


def _db_path(data_dir: str | Path, ticker: str) -> Path:
    """Return the SQLite checkpoint DB path for a ticker."""
    # Reject ticker values that would escape the checkpoints directory.
    safe = safe_ticker_component(ticker).upper()
    p = Path(data_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.db"


def thread_id(ticker: str, date: str) -> str:
    """Deterministic thread ID for a ticker+date pair."""
    return hashlib.sha256(f"{ticker.upper()}:{date}".encode()).hexdigest()[:16]


@contextmanager
def get_checkpointer(data_dir: str | Path, ticker: str) -> Generator[SqliteSaver, None, None]:
    """Context manager yielding a SqliteSaver backed by a per-ticker DB."""
    db = _db_path(data_dir, ticker)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def has_checkpoint(data_dir: str | Path, ticker: str, date: str) -> bool:
    """Check whether a resumable checkpoint exists for ticker+date."""
    return checkpoint_step(data_dir, ticker, date) is not None


def checkpoint_step(data_dir: str | Path, ticker: str, date: str) -> int | None:
    """Return the step number of the latest checkpoint, or None if none exists."""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return None
    tid = thread_id(ticker, date)
    with get_checkpointer(data_dir, ticker) as saver:
        config = {"configurable": {"thread_id": tid}}
        cp = saver.get_tuple(config)
        if cp is None:
            return None
        return cp.metadata.get("step")


def clear_all_checkpoints(data_dir: str | Path) -> int:
    """Remove all checkpoint DBs. Returns number of files deleted."""
    cp_dir = Path(data_dir) / "checkpoints"
    if not cp_dir.exists():
        return 0
    dbs = list(cp_dir.glob("*.db"))
    for db in dbs:
        db.unlink()
    return len(dbs)


def clear_checkpoint(data_dir: str | Path, ticker: str, date: str) -> None:
    """Remove checkpoint for a specific ticker+date by deleting the thread's rows."""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return
    tid = thread_id(ticker, date)
    conn = sqlite3.connect(str(db))
    try:
        for table in ("writes", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()

    # Also remove the execution log.
    log_p = _log_path(data_dir, ticker, date)
    try:
        log_p.unlink(missing_ok=True)
    except Exception:
        pass
