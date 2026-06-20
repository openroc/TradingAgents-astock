"""Model pricing store — loads and manages per-model billing rates from models.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "userdata"
_MODELS_FILE = _DATA_DIR / "models.json"

# In-memory cache
_models_cache: dict[str, dict] = {}
_cache_loaded: bool = False


def _load() -> dict[str, dict]:
    """Load models.json into memory. Returns the models dict keyed by model name."""
    global _models_cache, _cache_loaded
    if _cache_loaded:
        return _models_cache

    if not _MODELS_FILE.exists():
        logger.warning("models.json not found at %s — billing will use defaults", _MODELS_FILE)
        _cache_loaded = True
        return {}

    try:
        data = json.loads(_MODELS_FILE.read_text(encoding="utf-8"))
        _models_cache = data.get("models", {})
        logger.info("Loaded pricing for %d models from models.json", len(_models_cache))
    except Exception as exc:
        logger.warning("Failed to load models.json: %s", exc)
        _models_cache = {}

    _cache_loaded = True
    return _models_cache


def get_model_pricing(model_name: str) -> dict:
    """Return pricing dict for a model, or a default fallback.

    Returns: {name, provider, input_price_per_million, output_price_per_million, weight}
    """
    models = _load()
    if model_name in models:
        return models[model_name]
    logger.warning("Model '%s' not in models.json — using default pricing", model_name)
    return {
        "name": model_name,
        "provider": "unknown",
        "input_price_per_million": 2.10,
        "output_price_per_million": 8.40,
        "weight": 1.0,
    }


def list_models() -> list[dict]:
    """Return all models as a list sorted by provider then name."""
    models = _load()
    return sorted(models.values(), key=lambda m: (m.get("provider", ""), m.get("name", "")))


def update_model_pricing(model_name: str, updates: dict) -> bool:
    """Update pricing for a specific model in models.json.

    updates may contain: input_price_per_million, output_price_per_million, weight.
    Returns True if successful, False if the model was not found.
    """
    models = _load()
    if model_name not in models:
        return False

    # Validate
    for key in ("input_price_per_million", "output_price_per_million", "weight"):
        if key in updates:
            val = updates[key]
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"{key} must be a non-negative number, got {val!r}")

    models[model_name].update(updates)

    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _MODELS_FILE.write_text(
            json.dumps({"version": 1, "models": models}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Update in-memory cache
        global _models_cache
        _models_cache = models
        logger.info("Updated pricing for model '%s': %s", model_name, updates)
        return True
    except Exception as exc:
        logger.warning("Failed to write models.json: %s", exc)
        raise
