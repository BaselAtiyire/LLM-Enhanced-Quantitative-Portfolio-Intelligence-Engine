# 📊 LLM-Enhanced Quantitative Portfolio Intelligence Engine (HPIE)

> **A production-grade AI system combining LLM reasoning, multi-factor equity ranking, QLoRA fine-tuning, walk-forward backtesting, and role-based authentication in a single open-source deployable platform.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llm-enhanced-quantitative-portfolio-intelligence-engine-e385ca.streamlit.app/)

---

## 🌐 Live Demo

**URL:** https://llm-enhanced-quantitative-portfolio-intelligence-engine-e385ca.streamlit.app/

### 🔐 Demo Login
| Role | Username | Password |
|---|---|---|
| Demo (read-only) | `demo/admin` | `Demo@2026!` |

> Demo account has limited permissions. Backtest and Compare Runs panels require Admin access.

---

## 👨‍💻 Authors

**Basel Atiyire** (1st Author) — School of Computer Science, Western Illinois University  
**Ransom Igodi** (2nd Author) — School of Computer Science, Western Illinois University

---

## 📄 Research Paper

> B. Atiyire and R. Igodi, "A Hybrid LLM-Driven Portfolio Intelligence Engine for Explainable and Data-Driven Investment Decision Support," Draft Manuscript, April 2026.

**Key results (live yfinance data, 8 April 2026, 20-stock S&P 500 universe, 5-year period):**

| Metric | HPIE | Equal-Weight Benchmark | Delta |
|---|---|---|---|
| Sharpe Ratio | **1.109** | 0.883 | +0.226 |
| Annualised Return | **24.49%** | 18.70% | +5.79 pp |
| Annualised Volatility | 18.48% | 16.64% | +1.84 pp |
| Maximum Drawdown | -22.96% | -18.44% | -4.52 pp |

**Top-5 Rankings (8 April 2026):**

| Rank | Ticker | Score | 1W % | Sharpe | Ann Vol % |
|---|---|---|---|---|---|
| 1 | GOOGL | 0.697 | +11.34 | 0.681 | 30.43 |
| 2 | AAPL | 0.638 | +1.89 | 0.497 | 21.28 |
| 3 | NVDA | 0.626 | +6.32 | 1.160 | 37.78 |
| 4 | MSFT | 0.597 | +4.35 | 0.306 | 23.01 |
| 5 | AMZN | 0.589 | +7.24 | 0.211 | 29.08 |

---

## 🚀 Overview

HPIE is an end-to-end AI-powered quantitative decision intelligence platform that integrates:

- 🧠 **LLM reasoning** — Agno framework (GPT-4 / Groq LLaMA), quantitatively grounded
- 🤖 **QLoRA fine-tuning** — Mistral 7B fine-tuned on HPIE-generated instruction pairs
- 📈 **Multi-factor equity ranking** — 5 configurable factor-weight sliders
- 📊 **Risk modelling** — volatility, Sharpe ratio, drawdown, rolling volatility
- 🔄 **Walk-forward backtesting** — Sharpe-optimised, monthly rebalancing, benchmark comparison
- 💾 **Persistent storage** — SQLite run history with comparison engine
- 🔐 **Role-based authentication** — bcrypt (cost 12), Admin / User RBAC
- 💬 **Financial chatbot** — grounded, zero hallucination by design

---

## 🏗️ Architecture

The system follows a six-layer architecture:

| Layer | Panel / Module | Technology | Access |
|---|---|---|---|
| L1 | Controls — tickers, period, risk-free rate | yfinance, Streamlit | All users |
| L2 | Ranking weights — 5 sliders | NumPy / Pandas z-score engine | All users |
| L3 | AI Chatbot (Agno) — grounded | Agno + GPT-4 / Groq + QLoRA Mistral 7B | Sign-in required |
| L4 | Chat & Analysis — factor attribution | Additive score decomposition | All users |
| L5 | Compare Runs — SQLite history diff | SQLite snapshot engine | Admin only |
| L5 | Backtest — equity curve vs benchmark | Sharpe-optimised walk-forward | Admin only |
| L6 | Streamlit dashboard | Streamlit + Matplotlib | Role-gated |

---

## 🤖 QLoRA Fine-Tuning (C6)

The LLM reasoning agent has been fine-tuned using QLoRA on HPIE-generated instruction-following pairs:

- **Base model:** Mistral 7B Instruct v0.3
- **Method:** QLoRA — 4-bit NF4 quantisation + LoRA adapters (r=16, α=32)
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training:** 5 epochs, lr=2e-4, cosine schedule, Tesla T4 16GB GPU
- **Data:** Auto-generated from HPIE CSV exports via `generate_training_data.py`
- **Result:** Zero hallucinated values across all evaluated inference tests

### QLoRA Pipeline Files

| File | Purpose |
|---|---|
| `generate_training_data.py` | Converts HPIE CSV exports to Alpaca-format JSONL training pairs |
| `qlora_finetune.py` | Full QLoRA fine-tuning script (local GPU) |
| `HPIE_QLoRA_Colab.ipynb` | Google Colab notebook — runs on free T4 GPU |
| `integrate_qlora.py` | Loads fine-tuned adapter into HPIE app |

---

## 📈 Multi-Factor Ranking Engine

The scoring engine supports configurable weights:

```python
weights = {
    "market_cap":   0.30,   # higher is better
    "one_week_pct": 0.25,   # higher is better
    "ret_30d":      0.15,   # higher is better
    "trailing_pe":  0.15,   # lower is better (sign-inverted)
    "vol_ann":      0.15,   # lower is better (sign-inverted)
}
```

Composite score: `S_i = Σ w_k · z_{i,k}` where `z_{i,k} = (x_{i,k} - μ_k) / σ_k`

---

## 🔄 Walk-Forward Backtesting

- Rolling 252-day lookback window
- Monthly rebalancing
- Sharpe-optimised allocation (7,000 Dirichlet samples per rebalance)
- Equal-weight benchmark comparison
- Equity curve visualisation
- 5 bps one-way transaction costs

---

## 🔐 Authentication & Roles

| Feature | Unauthenticated | User | Admin |
|---|---|---|---|
| Controls & weights | ❌ | ✅ | ✅ |
| Run — generate dataset | ❌ | ✅ | ✅ |
| AI Chatbot | ❌ | ✅ | ✅ |
| Backtest panel | ❌ | Hidden | ✅ |
| Compare Runs panel | ❌ | Hidden | ✅ |

Passwords hashed with bcrypt (cost factor 12).

---

## ⚙️ Tech Stack

| Category | Technologies |
|---|---|
| Frontend | Streamlit |
| LLM Framework | Agno (GPT-4 / Groq LLaMA) |
| Fine-tuning | QLoRA — PEFT, bitsandbytes, TRL, Mistral 7B |
| Data | yfinance (Yahoo Finance) |
| Quant | NumPy, Pandas |
| Storage | SQLite |
| Auth | bcrypt, YAML |
| Visualisation | Matplotlib |
| Language | Python 3.10+ |

---

## 📂 Project Structure

```
LLM-Enhanced-Quantitative-Portfolio-Intelligence-Engine/
│
├── app_chatbot.py               # Main Streamlit application
├── reasoning_agent.py           # Quant engine + backtest + DB
├── generate_training_data.py    # QLoRA training data generator
├── qlora_finetune.py            # QLoRA fine-tuning script
├── integrate_qlora.py           # Load adapter into app
├── HPIE_QLoRA_Colab.ipynb      # Colab fine-tuning notebook
├── make_hash.py                 # bcrypt password hash generator
├── config.yaml                  # User credentials
├── requirements.txt
├── README.md
└── assets/
    └── architecture.png
```

---

## ▶️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/BaselAtiyire/LLM-Enhanced-Quantitative-Portfolio-Intelligence-Engine.git
cd LLM-Enhanced-Quantitative-Portfolio-Intelligence-Engine
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Authentication
```bash
python make_hash.py
# Copy hashes into config.yaml
```

### 5. Run Application
```bash
streamlit run app_chatbot.py
```

---

## 🤖 QLoRA Fine-Tuning Setup

### Step 1 — Export training data from your app
```bash
# Export CSV files from the app, then run:
python generate_training_data.py
# Produces: hpie_train.jsonl, hpie_eval.jsonl
```

### Step 2 — Fine-tune on Google Colab
- Upload `HPIE_QLoRA_Colab.ipynb` to Colab
- Set runtime to T4 GPU
- Upload the JSONL files when prompted
- Run all cells (~45 minutes)
- Download `hpie-llama-qlora.zip`

### Step 3 — Integrate into app
```bash
# Unzip adapter into project folder
# Set USE_QLORA=true in .env
python integrate_qlora.py  # test
```

---

## 🎯 Research Contributions

| # | Contribution |
|---|---|
| C1 | Six-layer architecture mapping directly to live UI panels |
| C2 | Quantitative grounding as a structural anti-hallucination mechanism |
| C3 | Configurable five-factor ranking with real-time weight adjustment |
| C4 | bcrypt role-based access control with admin-gated backtesting |
| C5 | Fully reproducible results from live yfinance data |
| C6 | QLoRA fine-tuning of Mistral 7B on HPIE-generated instruction pairs |

---

## 🚀 Future Work

- Convex optimisation (CVXPY) portfolio allocation
- Real-time news ingestion via RAG
- Online Bayesian weight updating from realised performance
- Docker containerisation
- MiFID II compliance certification pathway
- Multi-asset extension to fixed income and commodities
- Controlled user study measuring decision quality with/without HPIE

---

## 📌 Authors

**Basel Atiyire** — School of Computer Science, Western Illinois University  
**Ransom Igodi** — School of Computer Science, Western Illinois University

> *Building AI-powered financial intelligence systems.*
