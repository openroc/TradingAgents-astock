"""Per-user billing store — cumulative cost persists independently of tasks."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "userdata"


def _init_data_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Per-user store cache ───────────────────────────────────────────────────────

_store_cache: dict[str, BillingStore] = {}
_cache_lock = threading.Lock()


def _billing_file(user_id: str) -> Path:
    return _DATA_DIR / f"billing_{user_id}.json"


def get_billing_store(user_id: str) -> BillingStore:
    """Get or create the BillingStore for a given user (cached per process)."""
    with _cache_lock:
        if user_id not in _store_cache:
            _store_cache[user_id] = BillingStore(user_id)
        return _store_cache[user_id]


# ── BillingStore ────────────────────────────────────────────────────────────────

class BillingStore:
    """Thread-safe per-user billing store with JSON persistence.

    Costs are stored independently of tasks so that deleting a task does not
    affect the cumulative billing record.
    """

    def __init__(self, user_id: str):
        _init_data_dir()
        self._user_id = user_id
        self._file = _billing_file(user_id)
        self._lock = threading.Lock()
        self._cumulative_cost: float = 0.0
        self._records: list[dict] = []
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            with self._lock:
                self._cumulative_cost = data.get("cumulative_cost", 0.0)
                self._records = data.get("records", [])
            logger.info("Loaded billing for user %s: %.4f cumulative", self._user_id, self._cumulative_cost)
        except Exception as exc:
            logger.warning("Failed to load billing for %s: %s", self._user_id, exc)

    def _save(self) -> None:
        try:
            with self._lock:
                data = {
                    "cumulative_cost": self._cumulative_cost,
                    "records": self._records,
                }
            self._file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to persist billing for %s: %s", self._user_id, exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_task(
        self,
        task_id: str,
        ticker: str,
        tokens_in: int,
        tokens_out: int,
        model_name: str,
    ) -> float:
        """Record a completed task and return its computed cost.

        Pricing is looked up from models.json by model_name. The cost is added
        to the cumulative total. Returns the per-task cost.
        """
        from web_fastapi.model_pricing import get_model_pricing

        pricing = get_model_pricing(model_name)
        input_price = pricing["input_price_per_million"]
        output_price = pricing["output_price_per_million"]
        weight = pricing.get("weight", 1.0)

        input_cost = (tokens_in / 1_000_000) * input_price
        output_cost = (tokens_out / 1_000_000) * output_price
        task_cost = (input_cost + output_cost) * weight

        record = {
            "task_id": task_id,
            "ticker": ticker,
            "completed_at": time.time(),
            "model": model_name,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "input_price_per_million": input_price,
            "output_price_per_million": output_price,
            "weight": weight,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(task_cost, 6),
        }

        with self._lock:
            self._records.append(record)
            self._cumulative_cost += task_cost

        self._save()
        logger.info(
            "[%s] Billing recorded for user %s: tokens_in=%d tokens_out=%d cost=%.6f cumulative=%.4f",
            task_id, self._user_id, tokens_in, tokens_out, task_cost, self._cumulative_cost,
        )
        return round(task_cost, 6)

    def get_cumulative_cost(self) -> float:
        with self._lock:
            return round(self._cumulative_cost, 6)

    def get_records(self, page: int = 1, page_size: int = 20) -> dict:
        """Return paginated billing records (newest first)."""
        with self._lock:
            records = sorted(self._records, key=lambda r: r["completed_at"], reverse=True)
        total = len(records)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "cumulative_cost": round(self._cumulative_cost, 6),
            "total": total,
            "page": page,
            "page_size": page_size,
            "records": records[start:end],
        }
