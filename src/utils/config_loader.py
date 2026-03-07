from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "rubric" / "extraction_rules.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

if load_dotenv is not None:
    load_dotenv(dotenv_path=DEFAULT_ENV_PATH)


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config() -> dict[str, Any]:
    return {
        "triage_config": {
            "thresholds": {
                "scanned_image_char_density": 100,
                "mixed_origin_low_density_page_ratio": 0.25,
                "mixed_mode_ratio": 0.25,
                "high_font_density_char_count": 200,
                "table_heavy_density": 0.15,
                "single_column_min_chars_per_page": 1200,
                "single_column_min_bbox_density": 0.06,
                "multi_column_min_chars_per_page": 250,
                "multi_column_gutter_x_min": 200,
                "multi_column_gutter_x_max": 400,
                "multi_column_min_left_chars": 80,
                "multi_column_min_right_chars": 80,
                "multi_column_slice_count": 20,
                "multi_column_empty_slice_ratio": 0.8,
                "confidence_base": 0.9,
                "confidence_penalty_fonts_but_scan_like": 0.25,
                "confidence_penalty_no_fonts_but_text_rich": 0.3,
                "confidence_penalty_form_fillable_scan_like": 0.1,
            },
            "domain_keywords": {
                "financial": ["balance sheet", "income statement", "audit", "financial report", "revenue"],
                "legal": ["directive", "proclamation", "regulation", "court", "article"],
                "technical": ["standard", "specification", "architecture", "diagram"],
                "medical": ["patient", "diagnosis", "clinical", "treatment"],
                "technical_legal": ["vulnerability", "procedure", "compliance", "clause"],
            },
            "cost_tiers": {
                "strategy_a": 0.0,
                "strategy_b": 0.02,
                "strategy_c": 0.25,
            },
        },
        "chunking_constitution": {
            "max_chunk_size": 1000,
            "overlap_size": 200,
            "semantic_boundaries": [".\n", "\n\n", "###"],
        },
        "extraction_config": {
            "strategy_a": {
                "min_font_threshold": 1,
                "scanned_image_threshold": 0.35,
            },
            "strategy_c": {
                "api_key_env": "OPENROUTER_API_KEY",
                "model": "google/gemini-flash-1.5",
                "cost_per_1k_tokens": 0.000125,
                "cost_per_page": 0.01,
                "vlm_max_spend_per_doc": 0.5,
                "max_pages_to_process": 3,
            },
        },
        "router_config": {
            "escalation_thresholds": {
                "min_confidence_a": 0.8,
                "min_confidence_b": 0.7,
                "min_confidence_final": 0.65,
            }
        },
    }


@lru_cache(maxsize=8)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = default_config()
    path = _resolve_config_path(config_path)

    if not path.exists():
        return config

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        return config

    return _deep_merge(config, loaded)


def clear_config_cache() -> None:
    load_config.cache_clear()


def get_triage_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("triage_config", {})


def get_chunking_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("chunking_constitution", {})


def get_extraction_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("extraction_config", {})


def get_router_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("router_config", {})
