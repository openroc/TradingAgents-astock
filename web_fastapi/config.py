"""FastAPI application configuration — wraps DEFAULT_CONFIG with web-specific overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from tradingagents.default_config import DEFAULT_CONFIG


# Provider → default OpenAI-compatible base URL (None = use provider's official endpoint)
DEFAULT_BACKEND_URLS: dict[str, Optional[str]] = {
    "oneapi": "http://192.168.3.9:4000/v1",
    "openai": None,
    "deepseek": None,
    "qwen": None,
    "glm": None,
    "anthropic": None,
    "google": None,
    "azure": None,
    "ollama": None,
    "openrouter": None,
}


def build_config(
    llm_provider: str = "oneapi",
    deep_think_llm: Optional[str] = None,
    quick_think_llm: Optional[str] = None,
    backend_url: Optional[str] = None,
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
    output_language: str = "Chinese",
    data_vendors: Optional[dict] = None,
) -> dict:
    """Build a TradingAgents config dict for a specific task."""
    config = DEFAULT_CONFIG.copy()

    # Use provider's first listed model as default if not specified
    default_deep = deep_think_llm or (
        LLM_MODELS.get(llm_provider, ["gpt-4.1"])[0]
        if llm_provider in LLM_MODELS else "gpt-4.1"
    )
    default_quick = quick_think_llm or (
        LLM_MODELS.get(llm_provider, ["gpt-4.1-mini"])[0]
        if llm_provider in LLM_MODELS else "gpt-4.1-mini"
    )

    config.update({
        "llm_provider": llm_provider,
        "deep_think_llm": default_deep,
        "quick_think_llm": default_quick,
        "backend_url": backend_url or DEFAULT_BACKEND_URLS.get(llm_provider),
        "max_debate_rounds": max_debate_rounds,
        "max_risk_discuss_rounds": max_risk_discuss_rounds,
        "output_language": output_language,
    })
    if data_vendors:
        config["data_vendors"] = data_vendors
    return config


# Supported LLM providers for the UI dropdown
LLM_PROVIDERS = [
    "oneapi",
    "openai",
    "deepseek",
    "qwen",
    "glm",
    "anthropic",
    "google",
    "azure",
    "ollama",
    "openrouter",
]

LLM_MODELS: dict[str, list[str]] = {
    "oneapi": ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5", "MiniMax-M3"],
    "openai": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "qwen": ["qwen-plus", "qwen-max"],
    "glm": ["glm-4", "glm-4-flash"],
    "anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
    "google": ["gemini-2.0-flash", "gemini-2.5-pro"],
    "azure": ["gpt-4o"],
    "ollama": ["llama3.3", "qwen2.5"],
    "openrouter": ["openai/gpt-4o"],
}

OUTPUT_LANGUAGES = [
    "Chinese",
    "English",
]
