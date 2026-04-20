"""
ollama_integration.py — Ollama model integration for sentiment analysis

This module provides integration with locally running Ollama models
for sentiment analysis comparison with the traditional pipeline.
"""

from __future__ import annotations
import requests
import json
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Available Ollama models with their sizes (in GB) and recommended timeout
OLLAMA_MODELS = {
    "gemma3:4b": {"size": 3.3, "name": "Gemma 3 (4B)", "timeout": 120},
    "deepseek-r1:1.5b": {"size": 1.1, "name": "DeepSeek R1 (1.5B)", "timeout": 45},
    "qwen2.5:3b": {"size": 1.9, "name": "Qwen 2.5 (3B)", "timeout": 60},
    "qwen2.5:7b": {"size": 4.7, "name": "Qwen 2.5 (7B)", "timeout": 180},  # Increased to 180s (3 min)
    "zephyr:latest": {"size": 4.1, "name": "Zephyr", "timeout": 150},  # Increased to 150s
    "deepseek-coder:1.3b": {"size": 0.776, "name": "DeepSeek Coder (1.3B)", "timeout": 40},
    "qwen2:1.5b": {"size": 0.934, "name": "Qwen 2 (1.5B)", "timeout": 45},
    "tinyllama:latest": {"size": 0.637, "name": "TinyLlama", "timeout": 30},
    "qwen2:latest": {"size": 4.4, "name": "Qwen 2", "timeout": 150},  # Increased to 150s
    "phi3:latest": {"size": 2.2, "name": "Phi-3", "timeout": 60},
    "qwen2.5-coder:latest": {"size": 4.7, "name": "Qwen 2.5 Coder", "timeout": 180},  # Increased to 180s
}

# Estimated energy consumption based on model size (kWh per inference)
# Adjusted estimate: 0.00005 kWh per GB of model size (more realistic for local inference)
def estimate_ollama_energy(model_size_gb: float) -> float:
    """Estimate energy consumption based on model size"""
    base_energy = 0.00005  # kWh per GB (adjusted for local inference efficiency)
    return model_size_gb * base_energy


def get_ollama_models() -> List[Dict[str, str]]:
    """Get list of available Ollama models"""
    return [
        {"id": model_id, "name": info["name"], "size": info["size"]}
        for model_id, info in OLLAMA_MODELS.items()
    ]


def parse_sentiment_from_response(response_text: str) -> tuple[str, float, bool]:
    """
    Parse sentiment and confidence from Ollama model response.
    
    Returns:
        tuple: (label, confidence, success) where:
            - label is POSITIVE/NEGATIVE/NEUTRAL
            - confidence is float 0-1
            - success is True if confidence was extracted, False if not found
    """
    response_lower = response_text.lower()
    
    # Try to extract confidence score from model response
    confidence = None
    
    # Look for patterns like "Confidence: 85" or "confidence: 0.85" or "85%"
    import re
    
    # Pattern 1: "Confidence: XX" or "Confidence: 0.XX" (most flexible)
    conf_match = re.search(r'confidence[:\s]+(\d+\.?\d*)', response_lower)
    if conf_match:
        conf_value = float(conf_match.group(1))
        # Normalize to 0-1 range
        confidence = conf_value / 100 if conf_value > 1 else conf_value
    
    # Pattern 2: "XX%" format (only if not already found)
    if confidence is None:
        percent_match = re.search(r'(\d+\.?\d*)%', response_text)
        if percent_match:
            confidence = float(percent_match.group(1)) / 100
    
    # Pattern 3: Look for standalone numbers after sentiment (e.g., "NEGATIVE 85")
    if confidence is None:
        # Try to find a number that could be confidence (0-100 or 0.0-1.0)
        number_match = re.search(r'\b(\d+\.?\d*)\b', response_text)
        if number_match:
            num = float(number_match.group(1))
            # If it's between 0-100 or 0.0-1.0, assume it's confidence
            if 0 <= num <= 100:
                confidence = num / 100 if num > 1 else num
            elif 0 <= num <= 1:
                confidence = num
    
    # Look for explicit sentiment labels
    label = None
    if "positive" in response_lower:
        label = "POSITIVE"
    elif "negative" in response_lower:
        label = "NEGATIVE"
    elif "neutral" in response_lower:
        label = "NEUTRAL"
    
    # Check if we successfully extracted both label and confidence
    if label is None or confidence is None:
        # Return failure - model didn't follow instructions
        return "NEUTRAL", 0.0, False
    
    # Ensure confidence is in valid range
    confidence = max(0.0, min(1.0, confidence))
    
    return label, confidence, True


def run_ollama_inference(model_id: str, text: str, timeout: int = 120) -> Dict:
    """
    Run sentiment analysis using an Ollama model.
    
    Parameters
    ----------
    model_id : str
        Ollama model identifier (e.g., "gemma3:4b")
    text : str
        Input text for sentiment analysis
    timeout : int
        Request timeout in seconds
    
    Returns
    -------
    dict
        label        – "POSITIVE" | "NEGATIVE" | "NEUTRAL"
        confidence   – float 0-1
        model        – model identifier
        model_name   – human-readable model name
        latency_ms   – inference time in milliseconds
        energy_kwh   – estimated energy consumed
        co2_kg       – estimated CO₂ emissions
        raw_response – full model response
        error        – error message if failed
    """
    t0 = time.perf_counter()
    
    if model_id not in OLLAMA_MODELS:
        return {
            "label": "ERROR",
            "confidence": 0.0,
            "model": model_id,
            "model_name": "Unknown",
            "latency_ms": 0,
            "energy_kwh": 0,
            "co2_kg": 0,
            "error": f"Model {model_id} not found",
        }
    
    model_info = OLLAMA_MODELS[model_id]
    
    # Construct prompt for sentiment analysis with confidence request
    prompt = f"""You must analyze sentiment and provide ONLY these two lines:

Sentiment: POSITIVE
Confidence: 85

Or:

Sentiment: NEGATIVE
Confidence: 75

Or:

Sentiment: NEUTRAL
Confidence: 60

Now analyze this text: "{text}"

Remember: Output ONLY two lines in the exact format shown above. No explanations."""
    
    try:
        # Call Ollama API
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model_id,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract response text
        response_text = result.get("response", "").strip()
        
        # Parse sentiment and confidence
        label, confidence, parse_success = parse_sentiment_from_response(response_text)
        
        # If parsing failed, return error
        if not parse_success:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.warning(f"Ollama {model_id} failed to provide proper format. Response: {response_text}")
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "model": model_id,
                "model_name": model_info["name"],
                "latency_ms": latency_ms,
                "energy_kwh": 0,
                "co2_kg": 0,
                "error": "Model did not provide confidence score in expected format",
                "raw_response": response_text,
            }
        
        # Calculate metrics
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        energy_kwh = estimate_ollama_energy(model_info["size"])
        co2_kg = energy_kwh * 0.475  # Using same CO2 intensity as main system
        
        logger.info(f"Ollama {model_id} inference: {label} @ {confidence:.3f}")
        
        return {
            "label": label,
            "confidence": confidence,
            "model": model_id,
            "model_name": model_info["name"],
            "latency_ms": latency_ms,
            "energy_kwh": energy_kwh,
            "co2_kg": co2_kg,
            "raw_response": response_text,
        }
        
    except requests.exceptions.Timeout:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "label": "ERROR",
            "confidence": 0.0,
            "model": model_id,
            "model_name": model_info["name"],
            "latency_ms": latency_ms,
            "energy_kwh": 0,
            "co2_kg": 0,
            "error": "Request timeout",
        }
    
    except requests.exceptions.ConnectionError:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "label": "ERROR",
            "confidence": 0.0,
            "model": model_id,
            "model_name": model_info["name"],
            "latency_ms": latency_ms,
            "energy_kwh": 0,
            "co2_kg": 0,
            "error": "Cannot connect to Ollama. Is it running?",
        }
    
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.error(f"Ollama inference error: {e}")
        return {
            "label": "ERROR",
            "confidence": 0.0,
            "model": model_id,
            "model_name": model_info["name"],
            "latency_ms": latency_ms,
            "energy_kwh": 0,
            "co2_kg": 0,
            "error": str(e),
        }


def run_multiple_ollama_models(model_ids: List[str], text: str) -> List[Dict]:
    """
    Run sentiment analysis on multiple Ollama models.
    
    Parameters
    ----------
    model_ids : List[str]
        List of Ollama model identifiers (max 3)
    text : str
        Input text for sentiment analysis
    
    Returns
    -------
    List[dict]
        List of results from each model
    """
    if len(model_ids) > 3:
        logger.warning(f"Too many models selected ({len(model_ids)}). Using first 3.")
        model_ids = model_ids[:3]
    
    results = []
    for model_id in model_ids:
        # Get model-specific timeout from OLLAMA_MODELS dict
        timeout = OLLAMA_MODELS.get(model_id, {}).get("timeout", 120)
        result = run_ollama_inference(model_id, text, timeout=timeout)
        results.append(result)
    
    return results


def check_ollama_availability() -> bool:
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False
