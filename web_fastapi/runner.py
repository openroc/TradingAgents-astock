"""Background thread runner — executes TradingAgentsGraph for a given task_id."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_analysis(task_id: str, ticker: str, trade_date: str, config: dict) -> None:
    """Execute the full TradingAgents pipeline in a daemon thread."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from cli.stats_handler import StatsCallbackHandler
    from web_fastapi.task_store import task_store
    from web.progress import PIPELINE_STAGES

    task = task_store.get_task(task_id)
    if task is None:
        logger.error("[%s] Task not found", task_id)
        return

    task_store.update_status(task_id, "running")
    task.tracker.is_running = True
    task.tracker.mark_stage_active("market")

    stats = StatsCallbackHandler()

    try:
        graph = TradingAgentsGraph(
            debug=True,
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

        signal = graph.process_signal(
            last_chunk.get("final_trade_decision", "")
        )

        task.tracker.mark_complete(last_chunk, signal)
        task.tracker.is_running = False
        task.tracker.is_complete = True

        graph.ticker = ticker
        graph._log_state(trade_date, last_chunk)

        # Build report summary for persistence
        report_summary = {
            "signal": signal,
            "market_report": _strip(str(last_chunk.get("market_report", "")))[:500],
            "final_decision": _strip(str(last_chunk.get("final_trade_decision", "")))[:1000],
        }

        # Save to disk
        _save_report_to_disk(last_chunk, ticker, trade_date)

        # Update store and persist
        task_store.update_completed(
            task_id, task.tracker, report_summary, final_state=last_chunk
        )
        task_store.update_status(task_id, "completed")

        logger.info("[%s] Analysis completed — signal: %s", task_id, signal)

    except Exception as exc:
        logger.exception("[%s] Analysis failed", task_id)
        task.tracker.mark_error(str(exc))
        task.tracker.is_running = False
        task_store.update_status(task_id, "error", error=str(exc))


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


def _update_tracker(chunk: dict[str, Any], tracker) -> None:
    """Update ProgressTracker from a LangGraph stream chunk."""
    report_map = {
        "market_report": "market",
        "sentiment_report": "social",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "policy_report": "policy",
        "hot_money_report": "hot_money",
        "lockup_report": "lockup",
    }

    for report_key, stage_id in report_map.items():
        content = chunk.get(report_key, "")
        if content and tracker.stage_status(stage_id) != "done":
            tracker.mark_stage_done(stage_id, _strip(str(content)))

    dqs = chunk.get("data_quality_summary", "")
    if dqs and tracker.stage_status("quality_gate") != "done":
        tracker.mark_stage_done("quality_gate", str(dqs))

    debate = chunk.get("investment_debate_state")
    if debate and isinstance(debate, dict):
        judge = debate.get("judge_decision", "")
        if judge and tracker.stage_status("debate") != "done":
            tracker.mark_stage_done("debate", str(judge))

    trader_plan = chunk.get("trader_investment_plan", "")
    if trader_plan and tracker.stage_status("trader") != "done":
        tracker.mark_stage_done("trader", _strip(str(trader_plan)))

    risk = chunk.get("risk_debate_state")
    if risk and isinstance(risk, dict):
        risk_judge = risk.get("judge_decision", "")
        if risk_judge and tracker.stage_status("risk") != "done":
            tracker.mark_stage_done("risk", str(risk_judge))

    final = chunk.get("final_trade_decision", "")
    if final and tracker.stage_status("pm") != "done":
        tracker.mark_stage_done("pm", _strip(str(final)))


def _strip(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def start_analysis(task_id: str, ticker: str, trade_date: str, config: dict) -> threading.Thread:
    """Launch the analysis in a daemon thread. Returns the thread handle."""
    t = threading.Thread(
        target=run_analysis,
        args=(task_id, ticker, trade_date, config),
        daemon=True,
    )
    t.start()
    return t
