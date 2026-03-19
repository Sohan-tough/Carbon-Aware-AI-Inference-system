# 🌿 Carbon-Aware Adaptive AI Inference System

A production-quality **Green AI** demonstration that minimises energy consumption
and carbon emissions during NLP inference by intelligently routing requests
through a three-stage adaptive pipeline.

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
│  (0.000001 kWh)       │
└────────┬──────────────┘
         │ no clear match
         ▼
┌───────────────────────┐
│  Stage 2: DistilBERT  │ ← ~38% of BERT energy
│  (0.000120 kWh)       │ confidence ≥ 0.80 → done
└────────┬──────────────┘
         │ confidence < 0.80
         ▼
┌───────────────────────┐
│  Stage 3: BERT        │ ← full model, maximum quality
│  (0.000320 kWh)       │
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

> **Note on PyTorch**: The default `requirements.txt` pulls the standard
> `torch` wheel (~2 GB). For a CPU-only install you can replace it with:
> ```bash
> pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
> ```

### 4 — Run the dashboard

```bash
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501**.

---

## 🤖 Models Used

| Stage | Model | Size | Notes |
|-------|-------|------|-------|
| 2 | `distilbert-base-uncased-finetuned-sst-2-english` | ~255 MB | 40% smaller than BERT, 60% faster |
| 3 | `textattack/bert-base-uncased-SST-2` | ~440 MB | Full BERT fine-tuned on SST-2 |

Models are downloaded automatically from HuggingFace Hub on first run and
cached locally. Subsequent runs are fast.

---

## 📊 Dashboard Sections

| Section | Description |
|---------|-------------|
| **Input** | Text area + example sentence picker |
| **Prediction Result** | Label, confidence, stage used |
| **Energy Metrics** | kWh, CO₂, savings %, Green Score |
| **Visualisation** | Bar chart, pie chart, CO₂ timeline |
| **Inference Log** | Session history table |

---

## 🌍 Energy & Carbon Accounting

Energy constants are empirically estimated from CPU benchmark runs:

| Stage | Energy | CO₂ (@ 0.475 kg/kWh) |
|-------|--------|----------------------|
| Rule Engine | 0.000001 kWh | 0.000000475 kg |
| DistilBERT  | 0.000120 kWh | 0.000057 kg |
| BERT        | 0.000320 kWh | 0.000152 kg |

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
| `SMALL_MODEL_NAME` | DistilBERT | Stage 2 model |
| `LARGE_MODEL_NAME` | BERT-SST2 | Stage 3 model |
| `SMALL_MODEL_THRESHOLD` | `0.80` | Min confidence for Stage 2 to accept |
| `CO2_INTENSITY` | `0.475` | kg CO₂ / kWh (IEA world average) |

---

## 🎓 Educational Notes

This project demonstrates several important **Green AI** and **Ethical AI** principles:

1. **Progressive complexity** — Use the cheapest resource that gives a good answer.
2. **Energy transparency** — Show users exactly how much energy each decision costs.
3. **Carbon accountability** — Track and surface CO₂ emissions alongside accuracy.
4. **Model cascading** — A classic efficiency pattern applicable to any AI stack.
5. **Hardware-agnostic** — Runs entirely on CPU; no GPU required.

---

## 🧩 Extending the System

- **Add more stages** — Insert a medium-sized model between DistilBERT and BERT.
- **Domain-specific rules** — Expand `rule_engine.py` with task-specific lexicons.
- **Real-time power** — Enable CodeCarbon for hardware-accurate measurements.
- **Database backend** — Replace the in-memory log with SQLite / PostgreSQL.
- **API endpoint** — Wrap `inference_pipeline.run_pipeline()` in a FastAPI router.

---

## 📄 Licence

MIT — free to use, modify, and distribute for educational or commercial purposes.
