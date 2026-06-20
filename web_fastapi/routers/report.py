"""Report view and PDF download endpoints (auth required)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pathlib import Path

from web_fastapi.task_store import get_store
from web_fastapi import auth
from web_fastapi.auth import Session


_templates = Jinja2Templates(
    directory=Path(__file__).parent.parent / "templates"
)

router = APIRouter(prefix="/report", tags=["report"])


def _check_access(task_id: str, session) -> object:
    """Verify the session has access to the task. Returns the task."""
    task = get_store(session.user_id).get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id != session.user_id and not session.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    return task


def _to_str(val) -> str:
    if not val:
        return ""
    if isinstance(val, dict):
        return val.get("judge_decision", "") or str(val)
    return str(val)


@router.get("/{task_id}", response_class=HTMLResponse)
async def view_report(
    request: Request,
    task_id: str,
    session: Session = Depends(auth.get_current_user),
) -> HTMLResponse:
    """Render the full report page (server-side); JS polls for running task updates."""
    task = _check_access(task_id, session)

    state = task.final_state or getattr(task.tracker, "final_state", None)
    signal = task.tracker.signal or ""

    reports = {
        "market_report":         _to_str(state.get("market_report")) if state else "",
        "sentiment_report":      _to_str(state.get("sentiment_report")) if state else "",
        "news_report":           _to_str(state.get("news_report")) if state else "",
        "fundamentals_report":   _to_str(state.get("fundamentals_report")) if state else "",
        "policy_report":         _to_str(state.get("policy_report")) if state else "",
        "hot_money_report":      _to_str(state.get("hot_money_report")) if state else "",
        "lockup_report":         _to_str(state.get("lockup_report")) if state else "",
        "data_quality_summary":  _to_str(state.get("data_quality_summary")) if state else "",
        "investment_plan":       _to_str(state.get("investment_plan")) if state else "",
        "trader_investment_plan": _to_str(state.get("trader_investment_plan")) if state else "",
        "final_trade_decision":  _to_str(state.get("final_trade_decision")) if state else "",
        "risk_judge_decision":   _to_str(
            state.get("risk_debate_state", {}).get("judge_decision", "")
            if isinstance(state.get("risk_debate_state"), dict)
            else state.get("risk_debate_state", "")
        ) if state else "",
    }

    # Compute cost from model pricing
    task_cost = task.cost
    model_name = None
    input_price = None
    output_price = None
    if task_cost is None:
        try:
            from web_fastapi.model_pricing import get_model_pricing
            model_name = task.config.get("deep_think_llm", "MiniMax-M2.7")
            pricing = get_model_pricing(model_name)
            input_price = pricing["input_price_per_million"]
            output_price = pricing["output_price_per_million"]
            tokens_in = task.tracker.tokens_in or 0
            tokens_out = task.tracker.tokens_out or 0
            weight = pricing.get("weight", 1.0)
            task_cost = round(
                ((tokens_in / 1_000_000) * input_price +
                 (tokens_out / 1_000_000) * output_price) * weight,
                6,
            )
        except Exception:
            pass

    return _templates.TemplateResponse(
        request,
        "report_page.html",
        {
            "task_id": task_id,
            "ticker": task.ticker,
            "trade_date": task.trade_date,
            "status": task.status,
            "error": task.error,
            "signal": signal,
            "reports": reports,
            "elapsed": round(task.tracker.elapsed, 1),
            "llm_calls": task.tracker.llm_calls,
            "tool_calls": task.tracker.tool_calls,
            "tokens_in": task.tracker.tokens_in,
            "tokens_out": task.tracker.tokens_out,
            "task_cost": task_cost,
            "model_name": model_name,
            "input_price": input_price,
            "output_price": output_price,
        },
    )


@router.get("/{task_id}/data")
def report_data(
    task_id: str,
    session: Session = Depends(auth.get_current_user),
):
    """Return task status and report content as JSON (used by AJAX polling)."""
    task = _check_access(task_id, session)

    if task.status == "error":
        return {
            "status": "error",
            "ticker": task.ticker,
            "trade_date": task.trade_date,
            "error": task.error or "未知错误",
        }

    if task.status != "completed":
        return {
            "status": task.status,
            "ticker": task.ticker,
            "trade_date": task.trade_date,
            "error": task.error if task.status == "error" else None,
            "current_stage": task.tracker.current_stage,
            "completed_stages": task.tracker.completed_stages,
            "elapsed": round(task.tracker.elapsed, 1),
            "llm_calls": task.tracker.llm_calls,
            "tool_calls": task.tracker.tool_calls,
            "tokens_in": task.tracker.tokens_in,
            "tokens_out": task.tracker.tokens_out,
        }

    state = task.final_state or getattr(task.tracker, "final_state", None)
    if not state:
        raise HTTPException(status_code=500, detail="Report data not available")

    signal = task.tracker.signal or ""

    return {
        "status": "completed",
        "ticker": task.ticker,
        "trade_date": task.trade_date,
        "signal": signal,
        "elapsed": round(task.tracker.elapsed, 1),
        "llm_calls": task.tracker.llm_calls,
        "tool_calls": task.tracker.tool_calls,
        "tokens_in": task.tracker.tokens_in,
        "tokens_out": task.tracker.tokens_out,
        "cost": task.cost,
        "reports": {
            "market_report":        _to_str(state.get("market_report")),
            "sentiment_report":     _to_str(state.get("sentiment_report")),
            "news_report":          _to_str(state.get("news_report")),
            "fundamentals_report":  _to_str(state.get("fundamentals_report")),
            "policy_report":        _to_str(state.get("policy_report")),
            "hot_money_report":     _to_str(state.get("hot_money_report")),
            "lockup_report":        _to_str(state.get("lockup_report")),
            "data_quality_summary": _to_str(state.get("data_quality_summary")),
            "investment_plan":      _to_str(state.get("investment_plan")),
            "trader_investment_plan": _to_str(state.get("trader_investment_plan")),
            "final_trade_decision": _to_str(state.get("final_trade_decision")),
            "risk_judge_decision": _to_str(
                state.get("risk_debate_state", {}).get("judge_decision", "")
                if isinstance(state.get("risk_debate_state"), dict)
                else state.get("risk_debate_state", "")
            ),
        },
    }


@router.get("/{task_id}/markdown")
def download_markdown(
    task_id: str,
    session: Session = Depends(auth.get_current_user),
) -> Response:
    """Download the full analysis report as a Markdown file."""
    task = _check_access(task_id, session)
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    state = task.final_state or getattr(task.tracker, "final_state", None)
    if not state:
        raise HTTPException(status_code=500, detail="Report data not available")

    signal = task.tracker.signal or ""

    lines = [
        f"# XT 分析报告",
        f"",
        f"**标的**: {task.ticker}",
        f"**日期**: {task.trade_date}",
        f"**信号**: {signal}",
        f"",
        f"---",
        "",
    ]

    sections = [
        ("market_report", "技术分析报告"),
        ("sentiment_report", "市场情绪报告"),
        ("news_report", "新闻舆情报告"),
        ("fundamentals_report", "基本面报告"),
        ("policy_report", "政策分析报告"),
        ("hot_money_report", "游资追踪报告"),
        ("lockup_report", "解禁减持报告"),
    ]

    for key, title in sections:
        content = state.get(key, "")
        if content:
            lines.extend([f"## {title}", "", content, ""])

    if state.get("data_quality_summary"):
        lines.extend(["", "## 数据质量门控", "", state["data_quality_summary"], ""])
    if state.get("investment_plan"):
        lines.extend(["", "## 投资计划", "", state["investment_plan"], ""])
    if state.get("trader_investment_plan"):
        lines.extend(["", "## 交易方案", "", state["trader_investment_plan"], ""])
    if state.get("final_trade_decision"):
        lines.extend(["", "## 最终决策", "", state["final_trade_decision"], ""])

    markdown = "\n".join(lines)

    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{task.ticker}_{task.trade_date}.md"'
        },
    )


@router.get("/{task_id}/pdf")
def download_pdf(
    task_id: str,
    session: Session = Depends(auth.get_current_user),
) -> StreamingResponse:
    """Generate and download the report as a PDF."""
    task = _check_access(task_id, session)
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    state = task.final_state or getattr(task.tracker, "final_state", None)
    if not state:
        raise HTTPException(status_code=500, detail="Report data not available")

    signal = task.tracker.signal or ""

    try:
        from web.pdf_export import generate_pdf
        pdf_bytes = generate_pdf(state, task.ticker, task.trade_date, signal)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    filename = f"{task.ticker}_{task.trade_date}.pdf"

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
