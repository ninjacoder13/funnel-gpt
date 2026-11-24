# Funnel-GPT (Broken Funnel Debugger)

A lightweight, zero-PII command-line tool and visual report generator that uses a **from-scratch PyTorch GPT model** (based on Sebastian Raschka's *Build a Large Language Model (From Scratch)*) to debug product conversion funnels.

## How It Works

1. **Happy-Path Pretraining:** Trains a pure PyTorch GPT (`MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel`) exclusively on **converting** user event sequences to learn the statistical grammar of a healthy funnel.
2. **Time-Aware Event Tokenization:** Encodes each event as a 2-token pair `[DT_BUCKET, event_name]` (`DT_START`, `DT_INSTANT` <5s, `DT_SEC` <60s, `DT_MIN` <600s, `DT_LONG` >=600s) wrapped in `[START]` and `[END]`.
3. **Surprisal Scoring & Breakpoint Clustering:** Scores **abandoned** sessions using per-event cross-entropy loss against the validation 95th-percentile baseline to pinpoint the exact UX breakpoints killing conversion.
4. **Self-Contained HTML Report:** Outputs a single zero-dependency `report.html` with Executive Summary metrics, Top 5 Broken Funnel Transitions, and Color-Coded Session Traces.

## Quickstart

```bash
# 1. Install dependencies into a local virtual environment
uv sync --extra dev

# 2a. Option A: Fetch a real public e-commerce clickstream slice (REES46, ~30k sessions, no account needed)
uv run python data/fetch_real_data.py --mb 25 --out data/real_ecommerce_events.csv
uv run funnel-debug run data/real_ecommerce_events.csv --goal purchase --epochs 5 --out report_real.html

# 2b. Option B: Generate synthetic e-commerce funnel data (5,000 sessions, 3 planted UX bugs)
uv run python data/generate_sample.py
uv run funnel-debug run data/sample_events.csv --goal purchase --epochs 5 --out report.html

# 3. Run unit & integration tests
uv run pytest
```

## Generated Artifacts

Each `funnel-debug run` produces:

| Artifact | Purpose |
| :--- | :--- |
| `artifacts/tokenizer.json` | Learned event + time-delta vocabulary mapping tokens to integer IDs |
| `artifacts/model.pt` | Trained PyTorch `GPTModel` weights, model config, and P95 surprisal threshold |
| `artifacts/metrics.json` | Session counts, conversion rate, parameter count, epoch loss history, and P95 threshold |
| `report.html` | Self-contained visual diagnostic report (Executive Summary, Top 5 Broken Transitions, Color-Coded Session Traces) |

## Input CSV Schema

Any generic, zero-PII 3-column CSV:

| Column | Type | Example |
| :--- | :--- | :--- |
| `session_id` | string | `sess_00001` |
| `timestamp` | ISO-8601 datetime | `2026-01-15T10:00:00Z` |
| `event_name` | string | `add_to_cart` |
