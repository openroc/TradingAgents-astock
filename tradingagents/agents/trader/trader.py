"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]

        # Collect A-stock specific analyst reports
        policy_report = state.get("policy_report", "")
        hot_money_report = state.get("hot_money_report", "")
        lockup_report = state.get("lockup_report", "")

        # Build optional A-stock context block
        astock_context_parts = []
        if policy_report:
            astock_context_parts.append(f"Policy Analysis Report:\n{policy_report}")
        if hot_money_report:
            astock_context_parts.append(f"Hot Money / Capital Flow Report:\n{hot_money_report}")
        if lockup_report:
            astock_context_parts.append(f"Lockup Expiry / Insider Reduction Report:\n{lockup_report}")
        astock_context = "\n\n".join(astock_context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent specialising in A-share (China mainland) stocks. "
                    "Translate the Research Manager's investment plan into a concrete, executable "
                    "transaction proposal. You must factor in A-stock trading constraints:\n"
                    "- T+1 settlement: shares bought today cannot be sold until the next trading day\n"
                    "- Daily price limits: main board ±10%, STAR/ChiNext ±20%, ST stocks ±5%\n"
                    "- Minimum lot: 100 shares (main board) or 200 shares (STAR/ChiNext)\n"
                    "- Trading hours: 09:30-11:30, 13:00-15:00 Beijing time\n"
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "Be specific about entry price, stop loss, and position sizing.\n"
                    "（以上参数仅供技术研究参考，不构成投资建议）\n\n"
                    "**STRICT OUTPUT RULE**:\n"
                    "Your response must contain exactly two parts in this order:\n"
                    "PART 1 — Format Block (MUST come first, before ANY other text):\n"
                    "<format>\n"
                    "**Action**: <Buy/Hold/Sell>\n"
                    "**Reasoning**: <2-4 sentence concise action plan>\n"
                    "**Entry Price**: <price>元 (or <price1>-<price2>元 for range)\n"
                    "**Stop Loss**: <price>元\n"
                    "**Position Sizing**: <e.g. 30% of portfolio>\n"
                    "FINAL TRANSACTION PROPOSAL: **<BUY/HOLD/SELL>**\n"
                    "</format>\n"
                    "PART 2 — 详细分析 (optional, only after </format> above):\n"
                    "<detail>\n"
                    "Write detailed analysis, tables, risk factors, trading plans here.\n"
                    "</detail>\n"
                    "\n"
                    "VIOLATION EXAMPLES (these are ALL WRONG — do not do any of these):\n"
                    "❌ ANY text before <format>: '# 交易提案\\n\\n**Action**'\n"
                    "❌ Format fields after detail section: '...\\n## 分析\\n**Action**: BUY'\n"
                    "❌ Missing </format> closing tag\n"
                    "❌ Writing PART 2 content before PART 1\n"
                    "❌ Adding extra sections between format fields\n"
                    "\n"
                    "CORRECT example:\n"
                    "<format>\n"
                    "**Action**: BUY\n"
                    "**Reasoning**: 中芯国际突破150元阻力...回调至150-155元是良好建仓窗口。\n"
                    "**Entry Price**: 150-155元\n"
                    "**Stop Loss**: 134元\n"
                    "**Position Sizing**: 组合的10-15%\n"
                    "FINAL TRANSACTION PROPOSAL: **BUY**\n"
                    "</format>\n"
                    "<detail>\n"
                    "## 详细分析...\n"
                    "[tables, risk factors, trading plans]\n"
                    "</detail>"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts (including market, "
                    f"sentiment, news, fundamentals, policy, capital flow, and lockup/reduction "
                    f"specialists), here is an investment plan for {company_name}.\n\n"
                    f"{instrument_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    + (f"Additional A-Stock Analyst Context:\n{astock_context}\n\n" if astock_context else "")
                    + "Leverage these insights to craft a precise transaction proposal."
                    + get_language_instruction()
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
