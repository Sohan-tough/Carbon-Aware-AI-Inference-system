# 🌿 Carbon-Aware Adaptive AI Inference System

A production-quality **Green AI** demonstration that minimises energy consumption
and carbon emissions during NLP inference by intelligently routing requests
through a three-stage adaptive pipeline with **3-class sentiment analysis** 
(Positive, Negative, Neutral).

---

## 💡 Core Concept

Instead of blindly running the largest model every time, the system escalates
through three stages — only spending energy that the task actually requires.

```
User Input
    │
    ▼
┌───────────────────────┐
│  Stage 1: Rule Engine │ ← near-zero energy (keyword lookup)
│  (0.000001 kWh)       │ 🟢 Positive / 🔴 Negative / 🟡 Neutral
└────────┬──────────────┘
         │ no clear match
         ▼
┌───────────────────────┐
│  Stage 2: RoBERTa     │ ← ~38% of BERT energy
│  (0.000120 kWh)       │ confidence ≥ 0.80 → done
└────────┬──────────────┘
         │ confidence < 0.80
         ▼
┌───────────────────────┐
│  Stage 3: BERT        │ ← full model, maximum quality
│  (0.000320 kWh)       │ multilingual support
└───────────────────────┘
```

---

## 📁 Project Structure

```
carbon_aware_ai/
├── app.py                ← Streamlit dashboard (entry point)
├── inference_pipeline.py ← Three-stage pipeline orchestrator
├── rule_engine.py        ← Stage 1: keyword-based classifier
├── model_loader.py       ← Cached HuggingFace model loading
├── energy_tracker.py     ← Energy & CO₂ accounting + CodeCarbon
├── dashboard_utils.py    ← Plotly chart builders
├── config.py             ← Central configuration & constants
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```

---

## 🚀 Installation & Setup

### 1 — Clone / copy the project

```bash
mkdir carbon_aware_ai && cd carbon_aware_ai
# copy all project files here
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: The system now uses CPU-optimized PyTorch for better 
> compatibility. If you encounter space issues, the installation automatically
> uses the CPU-only version to save disk space.

> **Python Version**: Tested with Python 3.8+ (including Python 3.13)

### 4 — Run the dashboard

```bash
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501**.

---

## 🤖 Models Used

| Stage | Model | Size | Sentiment Classes | Notes |
|-------|-------|------|-------------------|-------|
| 2 | `cardiffnlp/twitter-roberta-base-sentiment-latest` | ~500 MB | 3-class (Pos/Neg/Neu) | RoBERTa optimized for social media text |
| 3 | `nlptown/bert-base-multilingual-uncased-sentiment` | ~670 MB | 3-class (Pos/Neg/Neu) | Multilingual BERT for global support |

**Key Updates:**
- ✅ **3-Class Sentiment**: Now supports Positive, Negative, and Neutral predictions
- ✅ **Multilingual Support**: Stage 3 model handles multiple languages
- ✅ **Modern Models**: Updated from SST-2 binary models to contemporary 3-class models
- ✅ **Social Media Optimized**: Stage 2 model trained on Twitter data for better real-world performance

Models are downloaded automatically from HuggingFace Hub on first run and
cached locally. Subsequent runs are fast.

---

## 📊 Dashboard Sections

| Section | Description |
|---------|-------------|
| **Input** | Text area + example sentence picker (includes neutral examples) |
| **Prediction Result** | Label (🟢 Positive/🔴 Negative/🟡 Neutral), confidence, stage used |
| **Energy Metrics** | kWh, CO₂, savings %, Green Score (0-100) |
| **Visualisation** | Energy bar chart, stage distribution pie, CO₂ timeline, Green Score gauge |
| **Inference Log** | Session history table with full audit trail |

### 🎨 **New UI Features:**
- **3-Color Coding**: Green (Positive), Red (Negative), Yellow (Neutral)
- **Keyword Highlighting**: Shows which rule engine keywords matched
- **Real-time Charts**: Interactive Plotly visualizations
- **Environmental Dashboard**: Comprehensive energy and carbon tracking

---

## 🌍 Energy & Carbon Accounting

Energy constants are empirically estimated from CPU benchmark runs:

| Stage | Energy | CO₂ (@ 0.475 kg/kWh) | Sentiment Support |
|-------|--------|----------------------|-------------------|
| Rule Engine | 0.000001 kWh | 0.000000475 kg | 3-class (Pos/Neg/Neu) |
| RoBERTa     | 0.000120 kWh | 0.000057 kg | 3-class (Pos/Neg/Neu) |
| BERT        | 0.000320 kWh | 0.000152 kg | 3-class (Pos/Neg/Neu) |

**Enhanced Rule Engine:**
- **Positive Keywords**: "great", "excellent", "amazing", "love", "perfect", etc.
- **Negative Keywords**: "bad", "terrible", "awful", "hate", "worst", etc.
- **Neutral Keywords**: "okay", "fine", "average", "normal", "decent", etc.

If **CodeCarbon** is installed, real hardware power counters are used instead.

### Green Score Formula

```
Green Score = round(100 × (1 − energy_used / energy_BERT))
```

A score of **100** means the Rule Engine fired (virtually free).
A score of **0** means BERT was used (maximum energy).

---

## ⚙️ Configuration (`config.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `SMALL_MODEL_NAME` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Stage 2 model (3-class) |
| `LARGE_MODEL_NAME` | `nlptown/bert-base-multilingual-uncased-sentiment` | Stage 3 model (multilingual) |
| `SMALL_MODEL_THRESHOLD` | `0.80` | Min confidence for Stage 2 to accept |
| `CO2_INTENSITY` | `0.475` | kg CO₂ / kWh (IEA world average) |

**New Configuration Options:**
- **3-Class Support**: All models now handle Positive, Negative, and Neutral
- **Multilingual Ready**: Large model supports multiple languages
- **Enhanced Examples**: Added neutral sentiment examples

---

## 🎓 Educational Notes

This project demonstrates several important **Green AI** and **Ethical AI** principles:

1. **Progressive complexity** — Use the cheapest resource that gives a good answer.
2. **Energy transparency** — Show users exactly how much energy each decision costs.
3. **Carbon accountability** — Track and surface CO₂ emissions alongside accuracy.
4. **Model cascading** — A classic efficiency pattern applicable to any AI stack.
5. **Hardware-agnostic** — Runs entirely on CPU; no GPU required.
6. **Inclusive Design** — 3-class sentiment analysis covers more real-world scenarios.
7. **Multilingual Support** — Global accessibility through multilingual models.
8. **Ethical AI Principles** — Implements transparency, fairness, and environmental responsibility.

### 🌟 **Key Learning Outcomes:**
- **Green AI Practices**: How to build environmentally conscious ML systems
- **Adaptive Inference**: Smart model selection based on input complexity  
- **Energy Optimization**: Practical techniques for reducing ML carbon footprint
- **Ethical AI Implementation**: Real-world application of AI ethics principles
- **System Architecture**: Building modular, maintainable ML pipelines

---

## 🧩 Extending the System

- **Add more stages** — Insert a medium-sized model between RoBERTa and BERT.
- **Domain-specific rules** — Expand `rule_engine.py` with task-specific lexicons.
- **Real-time power** — Enable CodeCarbon for hardware-accurate measurements.
- **Database backend** — Replace the in-memory log with SQLite / PostgreSQL.
- **API endpoint** — Wrap `inference_pipeline.run_pipeline()` in a FastAPI router.
- **Multi-language rules** — Add keyword sets for different languages.
- **Custom thresholds** — Dynamic confidence thresholds based on energy costs.
- **A/B testing** — Compare different model combinations and thresholds.

### 🚀 **Production Deployment Ideas:**
- **Cloud deployment** — Deploy on AWS/GCP/Azure with auto-scaling
- **Edge computing** — Run on mobile/IoT devices for offline inference
- **Microservices** — Split into separate services for each stage
- **Monitoring** — Add Prometheus metrics and Grafana dashboards
- **Load balancing** — Distribute requests across multiple instances

---

## 🆕 Recent Updates

### Version 2.0 - Neutral Sentiment & Enhanced Models
- ✅ **3-Class Sentiment Analysis**: Added support for Neutral sentiment alongside Positive/Negative
- ✅ **Updated Models**: Replaced SST-2 binary models with modern 3-class models
- ✅ **Multilingual Support**: Stage 3 now uses multilingual BERT for global accessibility
- ✅ **Enhanced Rule Engine**: Added neutral keywords and improved conflict resolution
- ✅ **Better UI**: Color-coded predictions (🟢🔴🟡) and enhanced visualizations
- ✅ **Improved Compatibility**: Fixed Python 3.13 compatibility and dependency issues

### Model Migration
| Old Model (v1.0) | New Model (v2.0) | Improvement |
|-------------------|-------------------|-------------|
| `distilbert-base-uncased-finetuned-sst-2-english` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3-class + social media optimized |
| `textattack/bert-base-uncased-SST-2` | `nlptown/bert-base-multilingual-uncased-sentiment` | 3-class + multilingual |

---

## 📄 Licence

MIT — free to use, modify, and distribute for educational or commercial purposes.
