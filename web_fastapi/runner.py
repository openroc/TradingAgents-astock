"""Background thread runner — executes TradingAgentsGraph for a given task_id."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_analysis(task_id: str, ticker: str, trade_date: str, config: dict) -> None:
    """Execute the full TradingAgents pipeline in a daemon thread."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from cli.stats_handler import StatsCallbackHandler
    from web_fastapi.task_store import get_store_for_task
    from web.progress import PIPELINE_STAGES

    store, task = get_store_for_task(task_id)
    if task is None:
        logger.error("[%s] Task not found", task_id)
        return

    store.update_status(task_id, "running")
    # Reload task after status update
    task = store.get_task(task_id)
    task.tracker.is_running = True
    task.tracker.mark_stage_active("market")

    stats = StatsCallbackHandler()

    try:
        graph = TradingAgentsGraph(
            debug=os.getenv("XT_DEBUG", "0") == "1",
            config=config,
            callbacks=[stats],
        )

        init_state = graph.propagator.create_initial_state(ticker, trade_date)
        graph_args = graph.propagator.get_graph_args(callbacks=[stats])

        last_chunk: dict[str, Any] = {}

        for chunk in graph.graph.stream(init_state, **graph_args):
            last_chunk = chunk
            _update_tracker(chunk, task.tracker)
            s = stats.get_stats()
            task.tracker.update_stats(
                s["llm_calls"], s["tool_calls"], s["tokens_in"], s["tokens_out"]
            )
            # Persist tracker state periodically so the UI sees live updates
            store.sync_tracker(task_id, task.tracker)

        signal = graph.process_signal(
            last_chunk.get("final_trade_decision", "")
        )

        # Convert all text fields to HTML-safe form for the web UI:
        # <think> tags removed, newlines → <br>
        _htmlify_chunk(last_chunk)

        task.tracker.mark_complete(last_chunk, signal)
        task.tracker.is_running = False
        task.tracker.is_complete = True

        graph.ticker = ticker
        graph._log_state(trade_date, last_chunk)

        report_summary = {
            "signal": signal,
            "market_report": (last_chunk.get("market_report") or "")[:500],
            "final_decision": (last_chunk.get("final_trade_decision") or "")[:1000],
        }

        _save_report_to_disk(last_chunk, ticker, trade_date)

        # Record billing (user pricing from users.json, fallback to models.json)
        task_cost = None
        try:
            from web_fastapi import user_store as user_store_module
            from web_fastapi.billing_store import get_billing_store
            from web_fastapi.user_store import get_user_by_id
            billing = get_billing_store(task.user_id)
            model_name = task.config.get("deep_think_llm", "unknown")
            user_store_module._load()  # reload so user pricing changes take effect without restart
            user = get_user_by_id(task.user_id)
            user_pricing = None
            if user and user.input_price_per_million and user.output_price_per_million:
                user_pricing = {
                    "input_price_per_million": user.input_price_per_million,
                    "output_price_per_million": user.output_price_per_million,
                }
            task_cost = billing.record_task(
                task_id=task_id,
                ticker=ticker,
                tokens_in=task.tracker.tokens_in,
                tokens_out=task.tracker.tokens_out,
                model_name=model_name,
                user_pricing=user_pricing,
            )
        except Exception as exc:
            logger.warning("[%s] Failed to record billing: %s", task_id, exc)

        store.update_completed(
            task_id, task.tracker, report_summary, final_state=last_chunk, cost=task_cost
        )
        store.update_status(task_id, "completed")

        logger.info("[%s] Analysis completed -- signal: %s", task_id, signal)

    except Exception as exc:
        logger.exception("[%s] Analysis failed", task_id)
        task.tracker.mark_error(str(exc))
        task.tracker.is_running = False

        # Record billing even for failed tasks (tokens were consumed)
        try:
            from web_fastapi import user_store as user_store_module
            from web_fastapi.billing_store import get_billing_store
            from web_fastapi.user_store import get_user_by_id
            billing = get_billing_store(task.user_id)
            model_name = task.config.get("deep_think_llm", "unknown")
            user_store_module._load()  # reload so user pricing changes take effect without restart
            user = get_user_by_id(task.user_id)
            user_pricing = None
            if user and user.input_price_per_million and user.output_price_per_million:
                user_pricing = {
                    "input_price_per_million": user.input_price_per_million,
                    "output_price_per_million": user.output_price_per_million,
                }
            billing.record_task(
                task_id=task_id,
                ticker=ticker,
                tokens_in=task.tracker.tokens_in,
                tokens_out=task.tracker.tokens_out,
                model_name=model_name,
                user_pricing=user_pricing,
            )
        except Exception as billing_exc:
            logger.warning("[%s] Failed to record billing for failed task: %s", task_id, billing_exc)

        store.update_status(task_id, "error", error=str(exc))


def _save_report_to_disk(state: dict, ticker: str, trade_date: str) -> None:
    """Save the complete report to disk (mirrors cli/main.py)."""
    try:
        import datetime
        from cli.main import save_report_to_disk

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path.cwd() / "reports" / f"{ticker}_{trade_date}_{timestamp}"
        save_report_to_disk(state, ticker, save_path)
        logger.info("Report saved to: %s", save_path)
    except Exception as exc:
        logger.warning("Failed to save report to disk: %s", exc)


def _update_tracker(chunk: dict, tracker) -> None:
    """Sync a graph chunk into the ProgressTracker.

    The graph streams chunks keyed by report field names (e.g. market_report,
    news_report, etc.) rather than explicit stage names. We infer the active
    stage by finding the first pipeline stage whose report_key is not yet
    present in the accumulated tracker state.
    """
    from web.progress import PIPELINE_STAGES

    # report_key -> stage_id reverse map
    _report_to_stage = {s["report_key"]: s["id"] for s in PIPELINE_STAGES}

    # Sync completed stages: a stage is complete when its report_key appears
    # in the chunk with a non-empty value.
    newly_completed = []
    for key, val in chunk.items():
        if key not in _report_to_stage:
            continue
        # Only mark string fields as complete if they are non-empty.
        # risk_debate_state is a dict; use its judge_decision sub-field as the
        # indicator of meaningful content.
        if isinstance(val, str):
            if not val.strip():
                continue
        elif isinstance(val, dict):
            judge = val.get("judge_decision")
            if not judge or not str(judge).strip():
                continue
        else:
            continue
        stage_id = _report_to_stage[key]
        if stage_id not in tracker.completed_stages:
            newly_completed.append(stage_id)

    for stage_id in newly_completed:
        tracker.completed_stages.append(stage_id)

    # Set current_stage to the first non-completed pipeline stage.
    completed_set = set(tracker.completed_stages)
    active = next(
        (s["id"] for s in PIPELINE_STAGES if s["id"] not in completed_set),
        "",
    )
    tracker.current_stage = active


def _strip_think_tags(text: str) -> str:
    """Strip LlamaIndex/LangChain debug blocks from LLM output (plain text, no HTML)."""
    import re
    return re.sub(r"\x03.*?\x03\s*", "", text, flags=re.DOTALL).strip()


def _html_text(text: str) -> str:
    """Strip <think> tags and convert newlines to <br> for HTML rendering.

    After <think> tags are stripped, bare newlines no longer produce visual breaks
    in HTML. Converting them to <br> ensures analyst reports, debate content, etc.
    display with correct paragraph structure in the web UI.
    """
    import re
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip control chars (e.g. SOH used by some LLM outputs) then convert \n → <br>
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.replace("\n", "<br>").strip()


def _htmlify_chunk(chunk: dict) -> None:
    """Recursively convert all string fields in a graph state chunk to HTML-safe text.

    Mutates chunk in place. Fields named "signal" and those used internally
    (node names, counters, etc.) are skipped.
    """
    skip_keys = {
        "signal", "node", "nodes", "count", "counter", "stage",
        "status", "ticker", "trade_date", "report_key", "id",
    }
    _htmlify_value(chunk, skip_keys)


def _htmlify_value(val: Any, skip_keys: set) -> None:
    if isinstance(val, dict):
        for k in list(val.keys()):
            if k in skip_keys or k.startswith("_"):
                continue
            v = val[k]
            if isinstance(v, str):
                val[k] = _html_text(v)
            else:
                _htmlify_value(v, skip_keys)
    elif isinstance(val, list):
        for i in range(len(val)):
            item = val[i]
            if isinstance(item, str):
                val[i] = _html_text(item)
            else:
                _htmlify_value(item, skip_keys)


def start_analysis(task_id: str, ticker: str, trade_date: str, config: dict) -> threading.Thread:
    """Launch the analysis in a daemon thread. Returns the thread handle."""
    t = threading.Thread(
        target=run_analysis,
        args=(task_id, ticker, trade_date, config),
        daemon=True,
    )
    t.start()
    return t
