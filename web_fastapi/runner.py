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
    from tradingagents.graph.checkpointer import get_execution_logger
    data_dir = config.get("data_cache_dir", os.path.expanduser("~/.tradingagents/cache"))
    exec_logger = get_execution_logger(data_dir, ticker, trade_date)

    try:
        graph = TradingAgentsGraph(
            debug=os.getenv("XT_DEBUG", "0") == "1",
            config=config,
            callbacks=[stats],
        )

        exec_logger.info(f"[ANALYSIS] Starting for {ticker} {trade_date}, task_id={task_id}")

        init_state = graph.propagator.create_initial_state(ticker, trade_date, task_id=task_id)
        graph_args = graph.propagator.get_graph_args(callbacks=[stats])

        last_chunk: dict[str, Any] = {}

        try:
            for chunk in graph.graph.stream(init_state, **graph_args):
                last_chunk = chunk
                _update_tracker(chunk, task.tracker)
                s = stats.get_stats()
                task.tracker.update_stats(
                    s["llm_calls"], s["tool_calls"], s["tokens_in"], s["tokens_out"]
                )
                # Persist tracker state periodically so the UI sees live updates
                store.sync_tracker(task_id, task.tracker)
        except Exception as exc:
            exec_logger.error(f"[ANALYSIS] Graph stream failed: {type(exc).__name__}: {exc}")
            raise

        exec_logger.info(f"[ANALYSIS] Completed for {ticker} {trade_date}")

        # Guard: PM stage must have produced a final_trade_decision.
        # If it's missing/empty, the task is incomplete — do NOT mark completed
        # so the checkpoint is preserved for resume.
        ftd = last_chunk.get("final_trade_decision")
        if not ftd or not str(ftd).strip():
            exec_logger.error("[ANALYSIS] final_trade_decision is empty — PM stage did not produce output. Raising error so task is not incorrectly marked completed.")
            raise RuntimeError(
                "Portfolio Manager did not produce final_trade_decision. "
                "Task left in error state with checkpoint preserved for resume."
            )

        signal = graph.process_signal(ftd)

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

        # Clear checkpoint after successful completion so next run starts fresh
        try:
            from tradingagents.graph.checkpointer import clear_checkpoint
            data_dir = config.get("data_cache_dir", os.path.expanduser("~/.tradingagents/cache"))
            clear_checkpoint(data_dir, ticker, trade_date)
        except Exception as exc:
            logger.warning("[%s] Failed to clear checkpoint: %s", task_id, exc)

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
        if "exec_logger" in dir():
            exec_logger.error(f"[ANALYSIS] FAILED: {type(exc).__name__}: {exc}")
        task.tracker.mark_error(str(exc))
        task.tracker.is_running = False

        # Record billing even for failed tasks (tokens were consumed)
        try:
            from web_fastapi import user_store as user_store_module
            from web_fastapi.billing_store import get_billing_store
            from web_fastapi.user_store import get_user_by_id
            billing = get_billing_store(task.user_id)
            model_name = task.config.get("deep_think_llm", "unknown")
            user_store_module._load()
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

    finally:
        if "exec_logger" in dir() and exec_logger is not None:
            exec_logger.close()


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
    """Strip <think> tags and control chars from text for HTML rendering.

    IMPORTANT: newlines are PRESERVED — do NOT convert \n to <br>.
    The browser-side marked.js with `breaks: true` handles newline-to-<br>
    conversion correctly. Converting to <br> server-side causes
    textContent → marked.parse() to fail, rendering markdown as plain text.
    """
    import re
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip control chars only; keep newlines for marked.js
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def _htmlify_chunk(chunk: dict) -> None:
    """Recursively convert all string fields in a graph state chunk to HTML-safe text.

    Mutates chunk in place. Fields named "signal" and those used internally
    (node names, counters, etc.) are skipped.
    """
    skip_keys = {
        "signal", "node", "nodes", "count", "counter", "stage",
        "status", "ticker", "trade_date", "report_key", "id",
        # Structured markdown outputs — skip so render_md sees clean newlines
        "investment_plan",
        "trader_investment_plan",
        # Nested markdown inside debate state dicts
        "bull_history",
        "bear_history",
        "aggressive_history",
        "conservative_history",
        "neutral_history",
        "judge_decision",
    }
    _htmlify_value(chunk, skip_keys)
    # Strip XML format tags from trader_investment_plan (not in skip_keys → handled above)
    if "trader_investment_plan" in chunk and isinstance(chunk["trader_investment_plan"], str):
        import re
        chunk["trader_investment_plan"] = re.sub(r"</?format>[\s\n]*", "", chunk["trader_investment_plan"], flags=re.IGNORECASE)
        chunk["trader_investment_plan"] = re.sub(r"</?detail>[\s\n]*", "", chunk["trader_investment_plan"], flags=re.IGNORECASE)


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


def resume_analysis(task_id: str, ticker: str, trade_date: str, config: dict) -> None:
    """Resume a task from its checkpoint. Uses graph.propagate() which auto-resumes."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.graph.checkpointer import get_execution_logger
    from cli.stats_handler import StatsCallbackHandler
    from web_fastapi.task_store import get_store_for_task
    from web.progress import PIPELINE_STAGES

    store, task = get_store_for_task(task_id)
    if task is None:
        logger.error("[%s] Task not found", task_id)
        return

    store.update_status(task_id, "running")
    task = store.get_task(task_id)
    task.tracker.is_running = True
    task.tracker.mark_stage_active("resume")

    stats = StatsCallbackHandler()
    data_dir = config.get("data_cache_dir", os.path.expanduser("~/.tradingagents/cache"))
    exec_logger = get_execution_logger(data_dir, ticker, trade_date)

    try:
        graph = TradingAgentsGraph(
            debug=os.getenv("XT_DEBUG", "0") == "1",
            config=config,
            callbacks=[stats],
        )

        exec_logger.info(f"[RESUME] Starting resume for {ticker} {trade_date}, task_id={task_id}")
        # propagate() handles checkpoint detection, error logging, and auto-resume;
        # it returns (final_state, signal).
        last_chunk, signal = graph.propagate(ticker, trade_date)
        exec_logger.info(f"[RESUME] Completed for {ticker} {trade_date}")

        # Guard: same check as run_analysis — PM must have produced output
        ftd = last_chunk.get("final_trade_decision")
        if not ftd or not str(ftd).strip():
            exec_logger.error("[RESUME] final_trade_decision is empty after resume. Raising error.")
            raise RuntimeError(
                "Portfolio Manager did not produce final_trade_decision on resume. "
                "Task left in error state."
            )

        _htmlify_chunk(last_chunk)

        task.tracker.mark_complete(last_chunk, signal)
        task.tracker.is_running = False
        task.tracker.is_complete = True

        report_summary = {
            "signal": signal,
            "market_report": (last_chunk.get("market_report") or "")[:500],
            "final_decision": (last_chunk.get("final_trade_decision") or "")[:1000],
        }

        _save_report_to_disk(last_chunk, ticker, trade_date)

        # Clear checkpoint after successful completion
        try:
            from tradingagents.graph.checkpointer import clear_checkpoint
            clear_checkpoint(data_dir, ticker, trade_date)
        except Exception as exc:
            logger.warning("[%s] Failed to clear checkpoint: %s", task_id, exc)

        task_cost = None
        try:
            from web_fastapi import user_store as user_store_module
            from web_fastapi.billing_store import get_billing_store
            from web_fastapi.user_store import get_user_by_id
            billing = get_billing_store(task.user_id)
            model_name = task.config.get("deep_think_llm", "unknown")
            user_store_module._load()
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

        logger.info("[%s] Resume completed -- signal: %s", task_id, signal)

    except Exception as exc:
        logger.exception("[%s] Resume failed", task_id)
        exec_logger.error(f"[RESUME] FAILED: {type(exc).__name__}: {exc}")
        task.tracker.mark_error(str(exc))
        task.tracker.is_running = False
        task.tracker.is_complete = True
        store.update_status(task_id, "error", error=str(exc))

    finally:
        exec_logger.close()


def start_resume_analysis(task_id: str, ticker: str, trade_date: str, config: dict) -> threading.Thread:
    """Launch the resume analysis in a daemon thread. Returns the thread handle."""
    t = threading.Thread(
        target=resume_analysis,
        args=(task_id, ticker, trade_date, config),
        daemon=True,
    )
    t.start()
    return t
