"""
model_comparison.py — Compare multiple models based on various metrics

This module provides utilities to compare traditional pipeline results
with Ollama model results and determine the best performing model.
"""

from __future__ import annotations
import pandas as pd
from typing import List, Dict, Optional


def calculate_model_score(result: Dict) -> float:
    """
    Calculate an overall score for a model based on multiple factors.
    
    Scoring criteria:
    - Confidence: 40%
    - Energy efficiency: 30%
    - Speed: 20%
    - Carbon footprint: 10%
    
    Returns a score between 0-100
    """
    # Normalize confidence (0-1 → 0-40)
    confidence_score = result.get("confidence", 0) * 40
    
    # Normalize energy (lower is better, 0.000320 kWh is baseline BERT)
    energy_kwh = result.get("energy_kwh", 0.000320)
    max_energy = 0.000320  # BERT energy as baseline
    energy_score = (1 - min(energy_kwh / max_energy, 1.0)) * 30
    
    # Normalize latency (lower is better, 5000ms as baseline)
    latency_ms = result.get("latency_ms", 5000)
    max_latency = 5000
    speed_score = (1 - min(latency_ms / max_latency, 1.0)) * 20
    
    # Normalize CO2 (lower is better, 0.000152 kg as baseline)
    co2_kg = result.get("co2_kg", 0.000152)
    max_co2 = 0.000152  # BERT CO2 as baseline
    carbon_score = (1 - min(co2_kg / max_co2, 1.0)) * 10
    
    total_score = confidence_score + energy_score + speed_score + carbon_score
    return round(total_score, 2)


def compare_models(traditional_result: Dict, ollama_results: List[Dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame for all models.
    
    Parameters
    ----------
    traditional_result : dict
        Result from the traditional pipeline (Rule Engine → RoBERTa → BERT)
    ollama_results : List[dict]
        Results from Ollama models
    
    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics
    """
    all_results = []
    
    # Add traditional pipeline result
    if traditional_result:
        all_results.append({
            "Model": f"Traditional ({traditional_result.get('stage', 'Unknown')})",
            "Label": traditional_result.get("label", "N/A"),
            "Confidence": f"{traditional_result.get('confidence', 0):.1%}",
            "Latency (ms)": f"{traditional_result.get('latency_ms', 0):.1f}",
            "Energy (kWh)": f"{traditional_result.get('energy_kwh', 0):.6f}",
            "CO₂ (kg)": f"{traditional_result.get('co2_kg', 0):.6f}",
            "Green Score": traditional_result.get("green_score", 0),
            "Overall Score": calculate_model_score(traditional_result),
        })
    
    # Add Ollama results
    for result in ollama_results:
        if result.get("error"):
            all_results.append({
                "Model": f"Ollama ({result.get('model_name', 'Unknown')})",
                "Label": "ERROR",
                "Confidence": "N/A",
                "Latency (ms)": f"{result.get('latency_ms', 0):.1f}",
                "Energy (kWh)": "N/A",
                "CO₂ (kg)": "N/A",
                "Green Score": 0,
                "Overall Score": 0,
            })
        else:
            # Calculate green score for Ollama model
            energy_kwh = result.get("energy_kwh", 0)
            # Green Score: percentage of energy saved compared to BERT (0.000320 kWh)
            bert_baseline = 0.000320
            green_score = max(0, round(100 * (1 - energy_kwh / bert_baseline), 1))
            
            all_results.append({
                "Model": f"Ollama ({result.get('model_name', 'Unknown')})",
                "Label": result.get("label", "N/A"),
                "Confidence": f"{result.get('confidence', 0):.1%}",
                "Latency (ms)": f"{result.get('latency_ms', 0):.1f}",
                "Energy (kWh)": f"{result.get('energy_kwh', 0):.6f}",
                "CO₂ (kg)": f"{result.get('co2_kg', 0):.6f}",
                "Green Score": green_score,
                "Overall Score": calculate_model_score(result),
            })
    
    df = pd.DataFrame(all_results)
    
    # Sort by Overall Score (descending)
    df = df.sort_values("Overall Score", ascending=False).reset_index(drop=True)
    
    return df


def get_best_model(traditional_result: Dict, ollama_results: List[Dict]) -> Dict:
    """
    Determine the best performing model based on overall score.
    
    Returns
    -------
    dict
        Best model information with reason
    """
    all_models = []
    
    # Add traditional result
    if traditional_result and not traditional_result.get("error"):
        all_models.append({
            "name": f"Traditional ({traditional_result.get('stage', 'Unknown')})",
            "score": calculate_model_score(traditional_result),
            "result": traditional_result,
        })
    
    # Add Ollama results
    for result in ollama_results:
        if not result.get("error"):
            all_models.append({
                "name": f"Ollama ({result.get('model_name', 'Unknown')})",
                "score": calculate_model_score(result),
                "result": result,
            })
    
    if not all_models:
        return {
            "name": "None",
            "score": 0,
            "reason": "No successful inferences",
        }
    
    # Find best model
    best = max(all_models, key=lambda x: x["score"])
    
    # Generate reason
    result = best["result"]
    reasons = []
    
    if result.get("confidence", 0) >= 0.85:
        reasons.append("high confidence")
    
    if result.get("energy_kwh", 1) < 0.0001:
        reasons.append("very low energy")
    elif result.get("energy_kwh", 1) < 0.0002:
        reasons.append("low energy")
    
    if result.get("latency_ms", 10000) < 500:
        reasons.append("fast response")
    
    reason = ", ".join(reasons) if reasons else "best overall balance"
    
    return {
        "name": best["name"],
        "score": best["score"],
        "reason": reason,
        "result": result,
    }
