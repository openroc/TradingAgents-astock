"""Report view and PDF download endpoints (auth required)."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pathlib import Path

from web_fastapi import auth
from web_fastapi.auth import Session
from web_fastapi.runner import _html_text  # strips think tags, preserves \n
import mistune


_templates = Jinja2Templates(
    directory=Path(__file__).parent.parent / "templates"
)

router = APIRouter(prefix="/report", tags=["report"])


def _check_access(task_id: str, session) -> object:
    """Verify the session has access to the task. Returns the task.

    Uses get_store_for_task to find the task across all user stores,
    then verifies the current session's user owns it (or is admin).
    """
    from web_fastapi.task_store import get_store_for_task

    store, task = get_store_for_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id != session.user_id and not session.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    return task


def _to_str(val) -> str:
    if not val:
        return ""
    if isinstance(val, dict):
        val = val.get("judge_decision", "") or str(val)
    return _html_text(str(val))


def _parse_final_decision(raw: str) -> dict:
    """Parse final_trade_decision markdown into structured sections for rich display."""
    import re

    if not raw:
        return {}

    # Extract Rating
    rating_match = re.search(r"\*?\*?Rating\*?\*?:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)

    # Extract Executive Summary — handle both markdown (**Executive Summary**:) and free-text (plain line at top)
    es_match = re.search(
        r"\*?\*?Executive Summary\*?\*?:\s*(.+?)(?=\n\*\*|\n【|\n*\[|\n*\*?\*?Rating|\n\nInvestment Thesis|\Z)",
        raw, re.DOTALL | re.IGNORECASE
    )
    es_match_content = None
    if not es_match:
        # Try plain text paragraph at very start (free-text format)
        lines = raw.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if line and len(line) > 30 and not line.startswith("**") and not line.startswith("Rating"):
                es_match_content = line
                break

    # Extract Investment Thesis section (works for both structured and free-text)
    it_start = raw.find("**Investment Thesis**")
    if it_start == -1:
        it_start = raw.find("Investment Thesis:")
    it_end = raw.find("【风险收益评估】")
    if it_end == -1:
        it_end = raw.find("Price Target:")
    investment_thesis = raw[it_start:it_end] if it_start != -1 else (raw if it_start == -1 else "")

    # Extract risk reward section
    rr_start = raw.find("【风险收益评估】")
    if rr_start == -1:
        rr_start = raw.find("【风险")
    rr_end = raw.find("**时间窗口", rr_start)
    if rr_end == -1:
        rr_end = raw.find("Price Target:", rr_start)
    risk_reward = raw[rr_start:rr_end] if rr_start != -1 else ""

    # Extract Price Target — handle both **Price Target**: and free-text "Price Target:"
    pt_match = re.search(r"Price Target\*?\*?:\s*([\d.]+)", raw, re.IGNORECASE)
    if not pt_match:
        pt_match = re.search(r"目标价\*?\*?:\s*([\d.]+)", raw, re.IGNORECASE)

    # Extract bull/bear sections (structured format)
    bull_factors = []
    bear_factors = []
    bull_start = investment_thesis.find("**支撑持仓")
    bear_start = investment_thesis.find("**抑制追高")

    if bull_start != -1 and bear_start != -1:
        bull_text = investment_thesis[bull_start:bear_start]
        for line in bull_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("**") or "支撑持仓" in line:
                continue
            m = re.match(r"^[-•*]\s+(.+)", line)
            if m and len(m.group(1)) > 5:
                bull_factors.append(m.group(1).strip())
                continue
            m2 = re.match(r"^[一二三四五六七八]、\s*(.+)", line)
            if m2 and len(m2.group(1)) > 5:
                bull_factors.append(m2.group(1).strip())

        bear_text = investment_thesis[bear_start:]
        for line in bear_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("**") or "抑制追高" in line:
                continue
            m = re.match(r"^[-•*]\s+(.+)", line)
            if m and len(m.group(1)) > 5:
                bear_factors.append(m.group(1).strip())
                continue
            m2 = re.match(r"^[一二三四五六七八]、\s*(.+)", line)
            if m2 and len(m2.group(1)) > 5:
                bear_factors.append(m2.group(1).strip())

    # Extract catalysts / time window
    tw_start = raw.find("**时间窗口")
    catalysts = []
    if tw_start != -1:
        tw_text = raw[tw_start:]
        for line in tw_text.split("\n"):
            line = line.strip()
            if not line or "时间窗口" in line or "Price Target" in line:
                continue
            m = re.match(r"^[-•*]\s+(.+)", line)
            if m and len(m.group(1)) > 5:
                catalysts.append(m.group(1).strip())

    # Parse thesis into structured sections (for free-text format with 【】 markers)
    thesis_sections = []
    if investment_thesis and not bull_factors and not bear_factors:
        import re as re2
        # Split by 【section headers】
        parts = re2.split(r'(?=【)', investment_thesis)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = re2.match(r'【(.+?)】', part)
            if m:
                section_title = m.group(1)
                section_body = part[len(m.group(0)):].strip()
                # Clean up Chinese number prefixes and bullet points
                lines = []
                for line in section_body.split('\n'):
                    line = line.strip()
                    if not line or len(line) < 5:
                        continue
                    line = re2.sub(r'^[一二三四五六七八九十]、\s*', '', line)
                    line = re2.sub(r'^\d+[.,、]\s*', '', line)
                    line = re2.sub(r'^[-•*]\s+', '', line)
                    if line:
                        lines.append(line)
                if lines:
                    # Determine section color
                    if '支持' in section_title or '做多' in section_title:
                        color = 'green'
                    elif '抑制' in section_title or '做空' in section_title or '风险' in section_title:
                        color = 'red'
                    elif '交易' in section_title or '策略' in section_title:
                        color = 'blue'
                    else:
                        color = 'gray'
                    thesis_sections.append({
                        'title': f"【{section_title}】",
                        'color': color,
                        'lines': lines,
                    })
            elif len(part) > 20:
                # Plain paragraph before any section header
                lines = []
                for line in part.split('\n'):
                    line = line.strip()
                    if line and len(line) > 10:
                        line = re2.sub(r'^[一二三四五六七八九十]、\s*', '', line)
                        line = re2.sub(r'^\d+[.,、]\s*', '', line)
                        if line:
                            lines.append(line)
                if lines:
                    thesis_sections.append({
                        'title': '',
                        'color': 'gray',
                        'lines': lines,
                    })

    # Rating badge color
    rating = rating_match.group(1).strip() if rating_match else "—"
    rating_upper = rating.upper()
    if "BUY" in rating_upper or "买入" in rating or "做多" in rating:
        rating_color = "buy"
    elif "SELL" in rating_upper or "卖出" in rating or "做空" in rating:
        rating_color = "sell"
    elif "HOLD" in rating_upper or "持有" in rating or "观望" in rating:
        rating_color = "hold"
    elif "OVERWEIGHT" in rating_upper or "增持" in rating:
        rating_color = "overweight"
    elif "UNDERWEIGHT" in rating_upper or "减持" in rating:
        rating_color = "underweight"
    else:
        rating_color = "neutral"

    return {
        "rating": rating,
        "rating_color": rating_color,
        "price_target": pt_match.group(1).strip() if pt_match else None,
        "executive_summary": (es_match.group(1).strip() if es_match else
                              (es_match_content.strip() if es_match_content else None)),
        "bull_factors": bull_factors,
        "bear_factors": bear_factors,
        "risk_reward_text": risk_reward.replace("【风险收益评估】", "").strip() if risk_reward else None,
        "catalysts": catalysts,
        "investment_thesis": (investment_thesis.replace("**Investment Thesis**:", "").strip()
                              if investment_thesis else None),
        "thesis_sections": thesis_sections,
        "has_structured": bool(
            (es_match and len(es_match.group(1)) > 50)
            or bull_factors
            or bear_factors
            or (risk_reward and len(risk_reward) > 20)
            or (investment_thesis and len(investment_thesis) > 100)
        ),
        "raw_content": raw.strip() if raw else None,
    }


def _parse_trader_plan(raw: str) -> dict:
    """Parse trader_investment_plan markdown into structured fields for rich display."""
    import re

    if not raw:
        return {}

    # Strip MiniMax/LLM thinking blocks before parsing
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

    action_match = re.search(r"\*\*Action\*\*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    entry_match = re.search(r"\*\*Entry Price\*\*:\s*([\d.]+)", raw, re.IGNORECASE)
    stop_match = re.search(r"\*\*Stop Loss\*\*:\s*([\d.]+)", raw, re.IGNORECASE)
    sizing_match = re.search(r"\*\*Position Sizing\*\*:\s*(.+?)(?=\n{2,}FINAL|\Z)", raw, re.DOTALL | re.IGNORECASE)
    # Fallback: extract sizing from **总仓位上限** table
    if not sizing_match:
        sizing_match = re.search(r"\*\*(?:总仓位上限|仓位上限)\*\*.*?\|\s*([^\n|]+)", raw)
    # Fallback: extract sizing from **上限** row: | **上限** | — | **70%** | 单票仓位上限约束 |
    if not sizing_match:
        sizing_match = re.search(r"\*\*上限\*\*.*?\|\s*—\s*\|\s*\*\*?([^\*]+)\*\*?", raw)
    final_match = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*\*?(.+?)\*\*?\s*$", raw, re.IGNORECASE | re.MULTILINE)
    reasoning_match = re.search(r"\*\*Reasoning\*\*:\s*(.+?)(?=\n\*\*Entry|\nFINAL|\Z)", raw, re.DOTALL | re.IGNORECASE)

    # Try to extract action from table rows (format: "买入/推荐" in table)
    if not action_match:
        action_match = re.search(r"(?:推荐操作|交易方向|委托方向|建议方向)[:：]\s*([^\s|]+)", raw)
    # Try to extract action from **建议方向** table field: | **建议方向** | 短线/波段做多（不推荐长持） |
    if not action_match:
        m = re.search(r"\*\*(?:建议方向)\*\*.*?\|\s*([^\n|]+)", raw)
        if m:
            action_match = m
    # Try from section heading: ### 2.1 交易方向：**做多（买入）**
    if not action_match:
        m = re.search(r"交易方向[：:]\*\*([^\*]+)\*\*", raw)
        if m:
            action_match = m
    # Fallback: scan tables for action keywords
    if not action_match:
        for kw in (r"做多", r"买入", r"开多", r"短线做多"):
            m = re.search(rf"\|.*{kw}.*\|", raw)
            if m:
                action_match = re.search(rf"({kw}[^\|]*)", m.group(0))
                break
    # Try to extract entry price from table cells
    if not entry_match:
        entry_match = re.search(r"(?:入场|建仓)(?:价格|价|区间)?[:：]\s*([\d.]+)", raw)
    # Try from **建仓区间** / **首次建仓** table fields
    if not entry_match:
        # Match | **建仓区间** | 135～145元（分批） | or | **首次建仓** | 20%仓位，约140元 |
        entry_match = re.search(r"\*\*(?:建仓区间|首次建仓)\*\*.*?\|\s*([\d.]+)", raw)
    # Try from **底仓** row: | **底仓** | 138-145元 | 30% |
    if not entry_match:
        entry_match = re.search(r"底仓.*?\|\s*([\d.]+)", raw)
    # Try to extract stop loss from table cells
    if not stop_match:
        stop_match = re.search(r"(?:止损)(?:价|位)?[:：]\s*([\d.]+)", raw)
    # Try from **硬止损** table field: | **硬止损** | **125元** |
    if not stop_match:
        stop_match = re.search(r"\*\*(?:硬止损|止损)\*\*.*?\*\*?([\d.]+)元", raw)
    # Try from **固定止损** table row: | **固定止损** | 125元 |
    if not stop_match:
        stop_match = re.search(r"固定止损.*?\|\s*([^元\n]+)元", raw)
    # Try bold-value table cells: | **建仓区间** | **35.00～36.00元** |
    if not entry_match:
        entry_match = re.search(r"\*\*(?:建仓区间|入场区间)\*\*.*?\|\s*\*\*([^\n]+?)\*\*", raw)
    # Try bold stop loss in table: | **硬止损（必须执行）** | **34.00元** | or | **止损** | **34.00元** |
    if not stop_match:
        stop_match = re.search(r"\*\*(?:硬止损|止损(?:位)?)\*\*.*?\|\s*\*\*([\d.]+)元", raw)
    # Try section heading with bold value: 入场区间：**35.00～36.00元**
    if not entry_match:
        entry_match = re.search(r"(?:入场|建仓)(?:区间)?[：:]\*\*([\d.]+)", raw)
    # Try stop loss in bold section: 止损位：**34.00元**
    if not stop_match:
        stop_match = re.search(r"止损(?:位|价)?[：:]\*\*([\d.]+)元", raw)

    action = action_match.group(1).strip().replace("**", "") if action_match else ""
    final_proposal = final_match.group(1).strip() if final_match else action

    action_upper = action.upper()
    if "BUY" in action_upper or "买入" in action or "做多" in action or "开多" in action:
        action_color = "buy"
    elif "SELL" in action_upper or "卖出" in action or "做空" in action or "平多" in action:
        action_color = "sell"
    elif "HOLD" in action_upper or "持有" in action or "观望" in action or "持仓" in action:
        action_color = "hold"
    else:
        action_color = "neutral"

    return {
        "action": action,
        "action_color": action_color,
        "final_proposal": final_proposal,
        "entry_price": entry_match.group(1).strip() if entry_match else None,
        "stop_loss": stop_match.group(1).strip() if stop_match else None,
        "position_sizing": sizing_match.group(1).strip().replace("\n", " ").replace("**", "") if sizing_match else None,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else None,
    }


def _render_field(text) -> str:
    """Strip <think> tags then render markdown. Used for all state fields that may contain LLM think blocks."""
    if not text:
        return ""
    return render_md(_html_text(text))


def render_md(text: str) -> str:
    """Render markdown text to HTML using mistune (server-side, no browser CDN needed).

    Single newlines in paragraphs are converted to <br>. Tables, headings, lists,
    blockquotes, and thematic breaks (---) are handled by mistune.
    """
    if not text:
        return ""
    import re
    # Pre-split thematic breaks (---/***/___) so each gets its own block
    # even when not preceded/followed by blank lines.
    text = re.sub(r"(?m)^([*-_]){3,}\s*$", "\n\n\\g<0>\n\n", text)
    lines_out = []
    for block in text.split("\n\n"):
        block = block.rstrip("\n")
        if not block:
            continue
        s = block.lstrip()
        if s.startswith("#"):
            lines_out.append(block)
        elif s.startswith("-") or s.startswith("*"):
            lines_out.append(block)
        elif s.startswith(">"):
            lines_out.append(block)
        elif block.startswith("```"):
            lines_out.append(block)
        elif "|" in block:
            lines_out.append(block)
        else:
            lines_out.append(block.replace("\n", "<br>"))
    return mistune.html("\n\n".join(lines_out))


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

    inv_debate = state.get("investment_debate_state", {}) if state else {}
    risk_debate = state.get("risk_debate_state", {}) if state else {}

    reports = {
        "market_report":          _render_field(state.get("market_report")) if state else "",
        "sentiment_report":       _render_field(state.get("sentiment_report")) if state else "",
        "news_report":            _render_field(state.get("news_report")) if state else "",
        "fundamentals_report":    _render_field(state.get("fundamentals_report")) if state else "",
        "policy_report":          _render_field(state.get("policy_report")) if state else "",
        "hot_money_report":       _render_field(state.get("hot_money_report")) if state else "",
        "lockup_report":          _render_field(state.get("lockup_report")) if state else "",
        "data_quality_summary":  _render_field(state.get("data_quality_summary")) if state else "",
        # Investment debate sub-fields (matches web: bull/bear history tabs)
        "bull_history":          _render_field(inv_debate.get("bull_history")) if isinstance(inv_debate, dict) else "",
        "bear_history":           _render_field(inv_debate.get("bear_history")) if isinstance(inv_debate, dict) else "",
        "trader_investment_plan": _render_field(state.get("trader_investment_plan")) if state else "",
        # Risk debate sub-fields (matches web: 4 sub-tabs)
        "risk_aggressive_history":    _render_field(risk_debate.get("aggressive_history")) if isinstance(risk_debate, dict) else "",
        "risk_conservative_history":  _render_field(risk_debate.get("conservative_history")) if isinstance(risk_debate, dict) else "",
        "risk_neutral_history":       _render_field(risk_debate.get("neutral_history")) if isinstance(risk_debate, dict) else "",
        "risk_judge_decision":   _render_field(
            risk_debate.get("judge_decision", "") if isinstance(risk_debate, dict) else ""
        ) if state else "",
        "final_trade_decision":  _render_field(state.get("final_trade_decision")) if state else "",
    }
    investment_plan = _render_field(state.get("investment_plan")) if state else ""

    # Parse final_trade_decision for rich display
    raw_ftd = _to_str(state.get("final_trade_decision")) if state else ""
    decision_parsed = _parse_final_decision(raw_ftd)

    # Parse trader_investment_plan for rich display
    raw_trader = _to_str(state.get("trader_investment_plan")) if state else ""
    trader_parsed = _parse_trader_plan(raw_trader)

    # Compute cost from model pricing
    task_cost = task.cost
    model_name = None
    input_price = None
    output_price = None
    if task_cost is None:
        try:
            from web_fastapi.model_pricing import get_model_pricing
            from web_fastapi.user_store import get_user_by_id
            model_name = task.config.get("deep_think_llm", "MiniMax-M2.7")
            tokens_in = task.tracker.tokens_in or 0
            tokens_out = task.tracker.tokens_out or 0
            user = get_user_by_id(session.user_id)
            if user and user.input_price_per_million and user.output_price_per_million:
                input_price = user.input_price_per_million
                output_price = user.output_price_per_million
                weight = 1.0
            else:
                pricing = get_model_pricing(model_name)
                input_price = pricing["input_price_per_million"]
                output_price = pricing["output_price_per_million"]
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
            "investment_plan": investment_plan,
            "final_trade_decision": _render_field(state.get("final_trade_decision")) if state else "",
            "decision_parsed": decision_parsed,
            "trader_parsed": trader_parsed,
            "trader_investment_plan": _render_field(state.get("trader_investment_plan")) if state else "",
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

    inv_debate = state.get("investment_debate_state", {})
    risk_debate = state.get("risk_debate_state", {})

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
        "investment_plan": _to_str(state.get("investment_plan")),
        "final_trade_decision": _to_str(state.get("final_trade_decision")),
        "trader_investment_plan": _to_str(state.get("trader_investment_plan")),
        "reports": {
            "market_report":           _to_str(state.get("market_report")),
            "sentiment_report":        _to_str(state.get("sentiment_report")),
            "news_report":             _to_str(state.get("news_report")),
            "fundamentals_report":     _to_str(state.get("fundamentals_report")),
            "policy_report":           _to_str(state.get("policy_report")),
            "hot_money_report":        _to_str(state.get("hot_money_report")),
            "lockup_report":           _to_str(state.get("lockup_report")),
            "data_quality_summary":   _to_str(state.get("data_quality_summary")),
            "bull_history":            _to_str(inv_debate.get("bull_history")) if isinstance(inv_debate, dict) else "",
            "bear_history":            _to_str(inv_debate.get("bear_history")) if isinstance(inv_debate, dict) else "",
            "risk_aggressive_history":    _to_str(risk_debate.get("aggressive_history")) if isinstance(risk_debate, dict) else "",
            "risk_conservative_history":  _to_str(risk_debate.get("conservative_history")) if isinstance(risk_debate, dict) else "",
            "risk_neutral_history":       _to_str(risk_debate.get("neutral_history")) if isinstance(risk_debate, dict) else "",
            "risk_judge_decision":    _to_str(
                risk_debate.get("judge_decision", "") if isinstance(risk_debate, dict) else ""
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

    inv_debate = state.get("investment_debate_state", {})
    risk_debate = state.get("risk_debate_state", {})

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

    # ── Part 1: 最终决策 (Portfolio Manager)
    if state.get("final_trade_decision"):
        lines.extend(["", "## 最终决策", "", state["final_trade_decision"], ""])

    # ── Part 2: 交易员决策 (Trader)
    if state.get("trader_investment_plan"):
        lines.extend(["", "## 交易员决策", "", state["trader_investment_plan"], ""])

    # ── Part 3: 研究经理结论 (Research Manager)
    if state.get("investment_plan"):
        lines.extend(["", "## 研究经理结论", "", state["investment_plan"], ""])

    # ── Part 4: 多空辩论
    if inv_debate.get("bull_history"):
        lines.extend(["", "### 多方历史", "", inv_debate["bull_history"], ""])
    if inv_debate.get("bear_history"):
        lines.extend(["", "### 空方历史", "", inv_debate["bear_history"], ""])

    # ── Part 5: 风控评估
    if risk_debate.get("aggressive_history"):
        lines.extend(["", "### 激进观点", "", risk_debate["aggressive_history"], ""])
    if risk_debate.get("conservative_history"):
        lines.extend(["", "### 保守观点", "", risk_debate["conservative_history"], ""])
    if risk_debate.get("neutral_history"):
        lines.extend(["", "### 中性观点", "", risk_debate["neutral_history"], ""])
    if risk_debate.get("judge_decision"):
        lines.extend(["", "### 风控决策", "", risk_debate["judge_decision"], ""])

    # ── Part 6: 分析师报告
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

    markdown = "\n".join(lines)

    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(f'{task.ticker}_{task.trade_date}.md')}"
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
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )
