# ProFit AI — Personalized Fitness with Reinforcement Learning

> Contextual Bandits + Thompson Sampling · GPT-4 AI Coach · Safety-Constrained RL · FastAPI · Streamlit

[![CI](https://github.com/your-username/RL/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/RL/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Results

> All metrics are from a **reproducible simulation** (N=1,000 episodes, seed=42, synthetic body-state data).
> Run yourself: `python scripts/benchmark.py --episodes 1000 --seed 42`

| Metric | Random Baseline | Rule-based | **Thompson Sampling** |
|--------|---------------:|----------:|----------------------:|
| Mean Reward | 0.601 | 0.618 | **0.643** |
| Std Reward | 0.229 | 0.218 | **0.200** |
| Optimal Action Rate | 8.9% | 5.4% | **18.0%** |
| Overtraining Rate | 0.6% | 0.0% | 0.7% |
| Late Mean Reward (ep 500–999) | 0.603 | 0.620 | **0.658** |

**+7.0%** cumulative reward over random · **+4.1%** over hand-crafted rules · converges at **~ep 84** · p99 API latency **<50ms**

---

## Problem & Why RL

Generic training programs ignore daily physiological variation. A plan suitable for a well-rested athlete is inappropriate — and potentially harmful — after poor sleep or high accumulated fatigue.

**Why Contextual Bandits instead of supervised learning?**

- No labelled dataset of "correct workouts" exists — feedback is implicit (completion, satisfaction)
- The reward signal arrives *after* the action, not before
- The action space is discrete (18 workout options) and safety-constrained
- Thompson Sampling gives Bayesian uncertainty estimates for free, enabling principled exploration without a separate exploration parameter

**Core design decisions:**

| Decision | Choice | Reason |
|----------|--------|--------|
| Algorithm | Beta-Bernoulli Thompson Sampling | Sample-efficient, closed-form Bayesian updates |
| Action space | 18 discrete actions (type × intensity × duration) | Clinically meaningful granularity |
| Safety layer | Hard-rule filter before bandit selection | RL must never recommend dangerous actions |
| Reward signal | Weighted composite (completion + adherence + recovery) | Aligns with real training outcomes |
| Online learning | Kafka-streamed feedback loop | Model improves continuously from real use |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User Layer                            │
│   Web UI (Streamlit)  ·  iOS App (future)  ·  API clients   │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│              API Gateway  (FastAPI)                          │
│         Authentication · Rate Limiting · Validation          │
└────────┬──────────────────────────────────────┬──────────────┘
         │                                      │
┌────────▼──────────────┐          ┌────────────▼─────────────┐
│  Recommendation Engine│          │     AI Coach Agent        │
│                       │          │                           │
│  1. Safety Gate       │          │  GPT-4 · Tool Calling     │
│     (hard rules)      │          │  Health data context      │
│  2. Feature extract   │          │  Conversational interface │
│  3. Thompson Sampling │          │                           │
│  4. Action selection  │          └───────────────────────────┘
└────────┬──────────────┘
         │
┌────────▼──────────────────────────────────────────────────────┐
│                       Data Layer                              │
│  Feature Store (Feast) · SQLite · Redis cache · Kafka queue   │
└────────┬──────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────┐
│              External Sources                                 │
│   Apple HealthKit · Oura Ring API v2 · OpenAI API             │
└───────────────────────────────────────────────────────────────┘
```

**Data flow:**
```
Wearable data / manual entry
  → 30+ engineered features (HRV trend, sleep debt, ACWR, rolling z-scores)
  → Safety Gate filters dangerous actions
  → Thompson Sampling selects from remaining actions
  → User completes (or skips) workout
  → Feedback streamed via Kafka
  → Beta parameters updated → better next recommendation
```

---

## Key Components

### 1. Thompson Sampling Contextual Bandit

`src/recommendation/contextual_bandits.py`

Beta-Bernoulli model over 18 discrete workout actions. Each action maintains independent Beta(α, β) parameters. At each step:

```python
# Sample from posterior for each allowed action
sample = np.random.beta(alpha[action_id], beta[action_id])

# Update after observing binary reward
alpha[action_id] += 1 if reward > 0.5 else 0
beta[action_id]  += 0 if reward > 0.5 else 1
```

Also implements `LinearContextualBandit` with full Bayesian linear regression posterior updates (B matrix, f vector) for continuous reward signals.

### 2. Safety-Constrained Action Filter

`src/safety/safety_gate.py`

Hard rules applied **before** bandit selection — RL cannot override these:

| Condition | Constraint |
|-----------|-----------|
| Readiness < 30 or Fatigue > 8 | REST or RECOVERY only |
| Fatigue > 6 | Max LOW intensity |
| 3+ consecutive high-load days | Max MEDIUM intensity |
| HRV below threshold | Restricted action space |

### 3. Feature Engineering Pipeline

`src/feature_store/feature_engineering.py`

30+ physiological features from raw wearable data:

- **Recovery**: HRV 7-day rolling mean, z-score, trend; sleep debt; resting HR deviation from baseline
- **Load**: Acute:Chronic Workload Ratio (ACWR), 7-day calorie/step sums
- **Consistency**: Training streak, completion rate, days since last session
- **Temporal**: Day-of-week, is_weekend (captures weekly periodicity)

### 4. Reward Function

`src/recommendation/reward_fn.py`

Multi-component weighted reward:

```
reward = 1.0 × completion
       + 0.5 × adherence_ratio
       - 1.0 × recovery_decline
       + 0.3 × satisfaction
       - 2.0 × overtraining_penalty
```

### 5. Online Learning Loop

`src/online_learning/loop.py`

Closed loop: state → recommendation → user feedback → Kafka event → Beta parameter update. Kafka is optional — system falls back gracefully to local event log.

### 6. GPT-4 AI Coach

`src/agent/coach_agent.py`

Three-layer architecture:
1. **Safety Gate** — blocks unsafe queries
2. **Recommendation Engine** — provides structured plan
3. **LLM Agent** — translates plan into natural language, handles Q&A

Tool calls available: `adjust_plan()`, `explain_plan()`, `mood_checkin()`, `set_micro_goal()`, `log_event()`.

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| RL Algorithm | Thompson Sampling (Beta-Bernoulli) | Workout recommendation |
| ML Framework | PyTorch | Model training |
| Feature Store | Feast | Feature management |
| Streaming | Apache Kafka | Online learning pipeline |
| API | FastAPI + Pydantic | Model serving (<50ms p99) |
| AI Coach | OpenAI GPT-4 | Conversational coaching |
| Web UI | Streamlit + Plotly | Interactive dashboard |
| Data Sources | Apple HealthKit, Oura API v2 | Wearable integration |
| Containerisation | Docker + Docker Compose | One-command deployment |
| CI/CD | GitHub Actions | Lint, test, security scan |
| Data Validation | Pydantic schemas | Input quality enforcement |
| Big Data (future) | PySpark | Multi-user scale-out |

---

## Quick Start

### Option A — Docker (all services, one command)

```bash
git clone https://github.com/your-username/RL.git && cd RL

# Configure
cp .env.example .env
# Edit .env: add OPENAI_API_KEY

# Launch (Web UI + API + Kafka + Redis)
docker-compose up

# Open
# Web interface → http://localhost:8501
# API docs      → http://localhost:8000/docs
```

### Option B — Local Python

```bash
pip install -r requirements.txt

cp .env.example .env   # add OPENAI_API_KEY

./start_web.sh         # starts API server + Streamlit
```

### Option C — Reproduce benchmark only (minimal deps)

```bash
pip install numpy scipy matplotlib
python scripts/benchmark.py --episodes 1000 --seed 42
# → scripts/benchmark_results.json
# → scripts/benchmark_learning_curve.png
```

---

## Project Structure

```
RL/
├── scripts/
│   ├── benchmark.py                 # ← Reproduce all reported metrics here
│   └── benchmark_results.json       # Last run results
├── src/
│   ├── recommendation/
│   │   ├── contextual_bandits.py    # Thompson Sampling (Beta + Linear)
│   │   ├── hybrid_recommender.py    # Rules + RL hybrid
│   │   ├── action_space.py          # 18 discrete workout actions
│   │   └── reward_fn.py             # Multi-component reward
│   ├── safety/
│   │   └── safety_gate.py           # Hard-rule action filter
│   ├── feature_store/
│   │   └── feature_engineering.py  # 30+ physiological features
│   ├── serving/
│   │   └── api_server.py            # FastAPI endpoints
│   ├── agent/
│   │   ├── coach_agent.py           # GPT-4 coach (3-layer)
│   │   ├── safety.py                # LLM safety guardrails
│   │   └── tools.py                 # Agent tool definitions
│   ├── online_learning/
│   │   ├── loop.py                  # Feedback → model update
│   │   └── kafka_consumer.py        # Streaming pipeline
│   ├── data_collection/
│   │   ├── apple_health.py
│   │   ├── oura_api.py
│   │   └── preprocess.py
│   ├── ab_testing/
│   │   └── experiment_framework.py
│   └── validation/
│       └── schemas.py               # Pydantic data schemas
├── web_app_pro.py                   # Streamlit UI (main)
├── Dockerfile                       # Multi-stage build
├── docker-compose.yml               # Full stack deployment
├── .github/workflows/ci.yml         # GitHub Actions CI
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Web Interface

[web_app_pro.py](web_app_pro.py) — Streamlit application with four tabs:

| Tab | What it does |
|-----|-------------|
| **Recommend** | Input today's body state → get RL recommendation → thumbs up/down feedback |
| **AI Coach** | GPT-4 chat with full health context; explains recommendations in plain English |
| **Analytics** | 7/14/30-day trends, HRV/sleep/fatigue correlation heatmap, training volume charts |
| **Settings** | User profile, historical data viewer, manual data entry, CSV/JSON upload |

Dark/Light mode toggle. No iOS developer account needed — manual data entry covers Apple Watch + Oura Ring values.

---

## Skills Demonstrated

### Machine Learning Engineering
- Bayesian RL (Beta-Bernoulli + Linear Thompson Sampling)
- Safety-constrained action selection (hard rules before RL)
- Multi-component reward design
- Online learning with incremental model updates
- Reproducible simulation benchmarking

### Software Engineering
- Production API design (FastAPI, Pydantic validation)
- Event-driven architecture (Kafka streaming)
- Feature store pattern (Feast)
- Containerisation with multi-stage Docker builds
- CI/CD pipeline (GitHub Actions: lint, test, security scan)

---

## Honest Limitations

| Claim | Reality |
|-------|---------|
| Benchmark metrics | Simulated environment, not real users |
| Kafka / Feast | Integrated in architecture; Kafka has local fallback |
| iOS integration | Data collection code written; no deployed app |
| `tests/` directory | Currently empty — unit tests are a known gap |

---

## Reproducing Results

```bash
# Exact command used to generate numbers in this README
python scripts/benchmark.py --episodes 1000 --seed 42
```

Raw output saved in [scripts/benchmark_results.json](scripts/benchmark_results.json).

---

## Future Work

- Unit + integration tests (highest priority)
- DQN / PPO for fine-grained exercise selection
- Multi-user support with collaborative filtering
- Native iOS app (HealthKit auto-sync)
- Model drift monitoring (Evidently AI)
- Cloud deployment (AWS/GCP + Kubernetes)

---

## License

MIT — see [LICENSE](LICENSE). Not a medical device. See license for full disclaimers.

---

## Contact

**Author**: [Your Name] · [LinkedIn] · [Email]

⭐ Star if useful · Issues welcome · PRs open
