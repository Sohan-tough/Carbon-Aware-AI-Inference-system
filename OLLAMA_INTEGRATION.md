# 🤖 Ollama Integration Guide

## Overview

The Carbon-Aware AI Inference System now supports **Ollama model integration**, allowing you to compare local LLM performance with the traditional pipeline (Rule Engine → RoBERTa → BERT).

## Features

✅ **Multi-Model Comparison** - Run up to 3 Ollama models simultaneously  
✅ **Side-by-Side Results** - Compare traditional pipeline with Ollama models  
✅ **Comprehensive Metrics** - Energy, CO₂, latency, confidence, and overall score  
✅ **Best Model Selection** - Automatically identifies the best performing model  
✅ **Real-time Inference** - All models run in parallel for fast results  

---

## Prerequisites

### 1. Install Ollama

Download and install Ollama from: https://ollama.ai

### 2. Pull Models

Pull the models you want to use:

```bash
ollama pull gemma3:4b
ollama pull tinyllama
ollama pull qwen2.5:3b
# ... etc
```

### 3. Start Ollama Server

```bash
ollama serve
```

The server runs on `http://localhost:11434` by default.

---

## Available Models

The system currently supports these Ollama models:

| Model | Size | Best For |
|-------|------|----------|
| **tinyllama:latest** | 637 MB | Fastest, lowest energy |
| **qwen2:1.5b** | 934 MB | Good balance |
| **deepseek-coder:1.3b** | 776 MB | Code-related sentiment |
| **qwen2.5:3b** | 1.9 GB | Better accuracy |
| **phi3:latest** | 2.2 GB | Microsoft's efficient model |
| **gemma3:4b** | 3.3 GB | Google's model |
| **zephyr:latest** | 4.1 GB | High quality |
| **qwen2.5:7b** | 4.7 GB | Best accuracy |

---

## How to Use

### Step 1: Select Ollama Models

1. In the right sidebar, you'll see **"🤖 Ollama Models"** dropdown
2. Select up to **3 models** you want to compare
3. If Ollama is not running, you'll see a warning

### Step 2: Enter Text

Enter your text in the input area (or select an example sentence)

### Step 3: Run Inference

Click **"⚡ Run Inference"** button

### Step 4: View Results

You'll see:

1. **Traditional Pipeline Result** - Rule Engine/RoBERTa/BERT result
2. **Ollama Models Results** - Results from each selected Ollama model
3. **Model Comparison Table** - Side-by-side comparison with metrics:
   - Label (Positive/Negative/Neutral)
   - Confidence
   - Latency (ms)
   - Energy (kWh)
   - CO₂ (kg)
   - Green Score
   - Overall Score
4. **Best Model** - Highlighted winner with reasoning

---

## Scoring System

Models are scored on a 0-100 scale based on:

- **Confidence** (40%) - How certain the model is
- **Energy Efficiency** (30%) - Lower energy = higher score
- **Speed** (20%) - Faster inference = higher score
- **Carbon Footprint** (10%) - Lower CO₂ = higher score

---

## Energy Estimation

Ollama model energy is estimated based on model size:

```
Energy (kWh) = Model Size (GB) × 0.0001
```

For example:
- **tinyllama** (637 MB) ≈ 0.0000637 kWh
- **gemma3:4b** (3.3 GB) ≈ 0.00033 kWh
- **qwen2.5:7b** (4.7 GB) ≈ 0.00047 kWh

---

## Example Comparison

**Input**: "This product is absolutely amazing!"

**Results**:

| Model | Label | Confidence | Latency | Energy | Overall Score |
|-------|-------|------------|---------|--------|---------------|
| Traditional (Rule Engine) | POSITIVE | 100% | 2ms | 0.000001 kWh | **98.5** 🏆 |
| Ollama (TinyLlama) | POSITIVE | 85% | 450ms | 0.000064 kWh | 87.2 |
| Ollama (Qwen 2.5 3B) | POSITIVE | 92% | 1200ms | 0.00019 kWh | 82.1 |

**Winner**: Traditional (Rule Engine) - high confidence, very low energy, fast response

---

## Troubleshooting

### "⚠️ Ollama not running"

**Solution**: Start Ollama server
```bash
ollama serve
```

### "Cannot connect to Ollama"

**Solution**: Check if Ollama is running on port 11434
```bash
curl http://localhost:11434/api/tags
```

### "Request timeout"

**Solution**: 
- Increase timeout in `ollama_integration.py`
- Use smaller models
- Check system resources

### Models not appearing

**Solution**: Pull the models first
```bash
ollama list  # Check installed models
ollama pull <model-name>  # Install missing models
```

---

## Tips for Best Results

1. **Start Small** - Try TinyLlama or Qwen 2:1.5b first
2. **Compare Wisely** - Select models of different sizes for meaningful comparison
3. **Watch Resources** - Larger models use more RAM and CPU
4. **Energy Conscious** - Smaller models are more energy-efficient
5. **Accuracy vs Speed** - Larger models are more accurate but slower

---

## Technical Details

### Files Added

- `ollama_integration.py` - Ollama API integration
- `model_comparison.py` - Model comparison logic
- `test_ollama.py` - Test script

### API Endpoint

```python
POST http://localhost:11434/api/generate
{
  "model": "tinyllama:latest",
  "prompt": "Analyze sentiment: ...",
  "stream": false
}
```

### Sentiment Parsing

The system parses Ollama responses to extract sentiment:
- Looks for keywords: "positive", "negative", "neutral"
- Adjusts confidence based on certainty words
- Defaults to neutral if unclear

---

## Future Enhancements

- [ ] Custom prompts for different models
- [ ] Model performance caching
- [ ] Batch inference support
- [ ] Custom energy profiles per model
- [ ] Model recommendation engine
- [ ] Historical comparison analytics

---

## Contributing

To add support for new Ollama models, update `OLLAMA_MODELS` in `ollama_integration.py`:

```python
OLLAMA_MODELS = {
    "your-model:tag": {"size": X.X, "name": "Your Model Name"},
}
```

---

## License

Same as main project (MIT)
