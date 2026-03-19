"""
config.py — Central configuration for Carbon-Aware AI Inference System
All constants, model names, and thresholds live here for easy tuning.
"""

# ── Model identifiers ──────────────────────────────────────────────────────────
# Using models that support 3-class sentiment (positive, negative, neutral)
SMALL_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LARGE_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# ── Inference thresholds ───────────────────────────────────────────────────────
RULE_ENGINE_THRESHOLD  = 1.0   # Rule engine always returns 1.0 when it fires
SMALL_MODEL_THRESHOLD  = 0.80  # DistilBERT must reach this to skip BERT
LARGE_MODEL_THRESHOLD  = 0.50  # BERT fallback — always accepted

# ── Approximate energy costs (kWh per inference) — empirical estimates ─────────
ENERGY_RULE_ENGINE   = 0.000001   # Essentially zero — pure Python logic
ENERGY_SMALL_MODEL   = 0.000120   # DistilBERT on CPU
ENERGY_LARGE_MODEL   = 0.000320   # BERT-base on CPU

# ── CO₂ intensity (kg CO₂ per kWh) — global average grid ──────────────────────
CO2_INTENSITY = 0.475  # kg CO₂ / kWh  (IEA world average 2023)

# ── Green Score weights (0–100) ───────────────────────────────────────────────
# Score = 100 × (1 − energy_used / energy_large_model)
GREEN_SCORE_MAX_ENERGY = ENERGY_LARGE_MODEL

# ── Streamlit page settings ───────────────────────────────────────────────────
PAGE_TITLE = "Carbon-Aware AI"
PAGE_ICON  = "🌿"
LAYOUT     = "wide"

# ── Example sentences for quick testing ───────────────────────────────────────
EXAMPLE_TEXTS = [
    "This product is absolutely amazing and exceeded all my expectations!",
    "The service was terrible and I want a full refund immediately.",
    "The package arrived on time and everything was in order.",
    "I have mixed feelings — some parts were great, others not so much.",
    "Worst experience ever. Total waste of money.",
    "It's okay, nothing special but does the job.",
    "The product is average, neither good nor bad.",
]
