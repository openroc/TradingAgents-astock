"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import logging

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**A-Stock Trading Constraints** (must factor into your decision):
- T+1 settlement: shares bought today cannot be sold until the next trading day
- Daily price limits: main board ±10%, STAR/ChiNext ±20%, ST stocks ±5%
- Minimum lot size: 100 shares (1 手) for main board; 200 shares for STAR/ChiNext
- Trading hours: 09:30-11:30, 13:00-15:00 (Beijing time)
- ST/delisting risk: ST or *ST status signals regulatory warning; factor into position sizing
- Margin eligibility: not all A-shares are margin-eligible; assume cash-only unless stated

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Required Output Format** (follow exactly for display on report page):
```
**Rating**: <Buy/Overweight/Hold/Underweight/Sell>

**Executive Summary**: <2-4 sentence concise action plan covering entry strategy, position sizing, key risk levels, and time horizon.>

**Investment Thesis**:
**支撑持仓**（做多理由，3-4点，每点以一、二、三、序号）:
一、<理由1>
二、<理由2>
三、<理由3>

**抑制追高**（做空/观望理由，3-4点，每点以一、二、三、序号）:
一、<理由1>
二、<理由2>
三、<理由3>

【风险收益评估】:
<2-3句话描述风险收益比、止损位、目标位>

**时间窗口**:
- <催化剂1>
- <催化剂2>
- <跟踪指标/验证节点>
```

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        # Attempt structured output first; fall back to free text on failure
        try:
            decision: PortfolioDecision | str = structured_llm.invoke(prompt)
        except Exception as exc:
            logger.warning(
                "Portfolio Manager: structured-output invocation failed (%s); "
                "retrying once as free text",
                exc,
            )
            response = llm.invoke(prompt)
            decision = getattr(response, "content", str(response))

        # Render the PM's final decision to markdown (used in PM tab and memory log)
        if isinstance(decision, PortfolioDecision):
            final_trade_decision = render_pm_decision(decision)
            # judge_decision: the PM's risk assessment of the debate,
            # derived from the structured fields — distinct from the full investment plan
            judge_decision = (
                f"**Rating**: {decision.rating.value}\n\n"
                f"**Investment Thesis**:\n{decision.investment_thesis}"
            )
        else:
            # Free-text fallback: two outputs are identical (no structured fields to split)
            final_trade_decision = decision
            judge_decision = decision

        new_risk_debate_state = {
            "judge_decision": judge_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
