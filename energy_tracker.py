"""
energy_tracker.py — Energy & carbon footprint estimation.

We provide two accounting approaches:

1. **Empirical lookup** (always available):
   Uses pre-measured per-stage energy constants from config.py.
   Instant, zero-overhead, works offline.

2. **CodeCarbon integration** (optional, soft-import):
   Wraps the inference call in a real hardware counter via the
   ``codecarbon`` library.  If the library is not installed the system
   falls back gracefully to the empirical lookup.

Public API
----------
estimate_energy(stage)   → dict with energy_kwh, co2_kg, green_score, saved_pct
measure_with_tracker(fn) → (result, emissions_data) using CodeCarbon if available
"""

from __future__ import annotations
import time
import logging
from typing import Callable, Any

from config import (
    ENERGY_RULE_ENGINE,
    ENERGY_SMALL_MODEL,
    ENERGY_LARGE_MODEL,
    CO2_INTENSITY,
    GREEN_SCORE_MAX_ENERGY,
)

logger = logging.getLogger(__name__)

# ── Per-stage energy lookup ────────────────────────────────────────────────────
_STAGE_ENERGY: dict[str, float] = {
    "Rule Engine":  ENERGY_RULE_ENGINE,
    "DistilBERT":   ENERGY_SMALL_MODEL,
    "BERT":         ENERGY_LARGE_MODEL,
}

# ── Optional CodeCarbon import ─────────────────────────────────────────────────
try:
    from codecarbon import EmissionsTracker as _CCTracker
    _CODECARBON_AVAILABLE = True
    logger.info("CodeCarbon found — hardware tracking enabled.")
except ImportError:
    _CODECARBON_AVAILABLE = False
    logger.info("CodeCarbon not installed — using empirical energy estimates.")


def estimate_energy(stage: str) -> dict:
    """
    Return energy/carbon metrics for the given inference stage.

    Parameters
    ----------
    stage : str
        One of ``"Rule Engine"``, ``"DistilBERT"``, ``"BERT"``.

    Returns
    -------
    dict
        energy_kwh   – kilowatt-hours consumed
        co2_kg       – kilograms of CO₂ equivalent
        green_score  – 0-100 environmental efficiency score
        saved_pct    – % energy saved vs always running BERT
        saved_kwh    – absolute kWh saved
    """
    energy_kwh = _STAGE_ENERGY.get(stage, ENERGY_LARGE_MODEL)
    co2_kg     = energy_kwh * CO2_INTENSITY

    # Green Score: 100 = as good as rule engine; 0 = ran full BERT
    green_score = max(
        0,
        round(100 * (1 - energy_kwh / GREEN_SCORE_MAX_ENERGY)),
    )
    green_score = min(green_score, 100)

    saved_kwh  = ENERGY_LARGE_MODEL - energy_kwh
    saved_pct  = round(100 * saved_kwh / ENERGY_LARGE_MODEL, 1)

    return {
        "energy_kwh":  energy_kwh,
        "co2_kg":      co2_kg,
        "green_score": green_score,
        "saved_pct":   max(saved_pct, 0.0),
        "saved_kwh":   max(saved_kwh, 0.0),
    }


def measure_with_tracker(fn: Callable[[], Any]) -> tuple[Any, dict]:
    """
    Execute *fn()* inside a CodeCarbon tracker (if available) and return
    ``(fn_result, emissions_dict)``.

    Falls back to empirical estimates when CodeCarbon is absent.

    Parameters
    ----------
    fn : Callable
        Zero-argument callable wrapping the inference pipeline.

    Returns
    -------
    tuple
        (result_of_fn, {energy_kwh, co2_kg, source})
    """
    if _CODECARBON_AVAILABLE:
        tracker = _CCTracker(
            project_name="carbon_aware_ai",
            measure_power_secs=1,
            log_level="error",
            save_to_file=False,
            allow_multiple_runs=True,
        )
        tracker.start()
        result = fn()
        emissions = tracker.stop()          # returns kg CO₂
        energy_kwh = emissions / CO2_INTENSITY if emissions else 0.0
        return result, {
            "energy_kwh": energy_kwh,
            "co2_kg":     emissions or 0.0,
            "source":     "CodeCarbon",
        }
    else:
        # No hardware counter — run the function and use lookup tables
        result = fn()
        return result, {"source": "empirical"}


def format_energy(kwh: float) -> str:
    """Human-readable kWh string with appropriate decimal places."""
    return f"{kwh:.6f} kWh"


def format_co2(kg: float) -> str:
    """Human-readable CO₂ string."""
    return f"{kg:.6f} kg CO₂"
