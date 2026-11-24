# Project Spec: Funnel-GPT (Broken Funnel Debugger)

## 1. Objective
Build a lightweight, zero-PII command-line tool and visual report generator that uses a **from-scratch PyTorch GPT model** (based on Sebastian Raschka's *Build a Large Language Model (From Scratch)*) to debug product conversion funnels.

Instead of predicting English text, the model trains exclusively on **converting** user event sequences ("happy paths") to learn the statistical grammar of a healthy funnel. At inference time, it scores **abandoned** sessions using per-event cross-entropy loss (surprisal) to pinpoint the exact UX breakpoints killing conversion.

## 2. Strict Constraints
- **Pure PyTorch GPT:** Implement `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, and `GPTModel` from scratch in PyTorch following Raschka's architecture. **Do NOT use Hugging Face `transformers`, `tiktoken`, or external LLM APIs.**
- **Zero-PII Input:** The tool must accept any generic 3-column CSV (`session_id`, `timestamp`, `event_name`).
- **Fast Local Execution:** Must auto-detect `mps` (Apple Silicon), `cuda`, or `cpu`, and train on 10k–50k sessions in under 3 minutes on a laptop.
- **Self-Contained Output:** Generate a single standalone `report.html` (no backend server required to view results).

## 3. Repository Structure
```text
funnel-gpt/
├── SPEC.md
├── pyproject.toml
├── data/
│   └── generate_sample.py    # Synthetic e-commerce funnel generator for instant testing
├── src/
│   ├── __init__.py
│   ├── tokenizer.py          # Event + time-delta discrete tokenizer
│   ├── dataset.py            # PyTorch Dataset & DataLoader (sliding window)
│   ├── model.py              # From-scratch Raschka GPTModel
│   ├── train.py              # Happy-path pretraining loop + checkpointing
│   ├── score.py              # Surprisal scoring + breakpoint clustering
│   ├── report.py             # Standalone HTML report generator
│   └── cli.py                # CLI entrypoint (`funnel-debug`)
└── tests/
    ├── test_tokenizer.py
    ├── test_model.py
    └── test_pipeline.py
```

## 4. Technical Specification

### 4.1 Input Schema & Synthetic Test Data

* **CSV Columns:** `session_id` (str), `timestamp` (ISO-8601 or parseable datetime), `event_name` (str).
* **Synthetic Generator (`data/generate_sample.py`):** Create a script that outputs `data/sample_events.csv` with 5,000 sessions:
  * 3,500 converting sessions following realistic paths (`page_view -> view_item -> add_to_cart -> begin_checkout -> add_shipping -> purchase`).
  * 1,500 abandoned sessions with 3 planted UX bugs (e.g., stuck in a `begin_checkout -> error_payment -> begin_checkout` loop, sudden `add_shipping -> page_view` bounce, or a 10-minute stall at `add_to_cart`).
  * This allows end-to-end verification before plugging in the real GA4 dataset.

### 4.2 Event + Time-Delta Tokenizer (`src/tokenizer.py`)

* Sort events chronologically per `session_id`.
* Compute `dt` (seconds since previous event in the same session) and bucket into discrete tokens:
  * `DT_START` (first event), `DT_INSTANT` (<5s), `DT_SEC` (<60s), `DT_MIN` (<600s), `DT_LONG` (>=600s).
* Represent each event as 2 tokens: `[DT_BUCKET, event_name]`.
* Wrap each session with `[START]` and `[END]`. Include `[PAD]` and `[UNK]`.
* Example sequence:
  `[START] DT_START page_view DT_SEC view_item DT_MIN add_to_cart DT_SEC purchase [END]`
* Support `save(path)` and `load(path)` as JSON.

### 4.3 Model Architecture (`src/model.py`)

Implement Raschka's `GPTModel` verbatim with this default mini configuration:

```python
MINI_GPT_CONFIG = {
    "vocab_size": None,      # Dynamically set from EventTokenizer
    "context_length": 128,   # Fits ~60 events per session
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

### 4.4 Training Pipeline (`src/train.py`)

* Filter input CSV to **converting sessions only** (sessions containing `--goal`, e.g., `purchase`).
* Split 90/10 into train/val converting sessions.
* Train with `AdamW` (`lr=3e-4`, `weight_decay=0.01`) using next-token cross-entropy loss (`ignore_index=pad_id`).
* Compute and store the **95th percentile per-event loss threshold** on the validation converting sessions (this serves as the baseline for "normal" surprisal).
* Save `model.pt`, `tokenizer.json`, and `metrics.json` to an output directory.

### 4.5 Surprisal Scorer & Breakpoint Clustering (`src/score.py`)

* Run all **non-converting sessions** through the trained model in `eval()` mode.
* For each event in a session, compute its surprisal: average cross-entropy loss of its `[DT_BUCKET, event_name]` pair.
* Identify the **breakpoint**: the first event whose surprisal exceeds the 95th percentile baseline (or the highest-surprisal event if none exceed it).
* Extract the **breakpoint transition** as `(prev_event -> DT_BUCKET -> breakpoint_event)`.
* Aggregate and rank the **Top 5 Broken Funnel Transitions** by frequency and average surprisal.

### 4.6 CLI & HTML Report (`src/cli.py`, `src/report.py`)

* **CLI Command:**
```bash
funnel-debug run data/sample_events.csv --goal purchase --epochs 10 --out report.html
```

* **HTML Report (`report.html`):** Clean, modern, zero-dependency HTML/CSS report containing:
  1. **Executive Summary:** Total sessions, conversion rate, model parameter count, final train/val loss, and 95th-percentile surprisal threshold.
  2. **Top 5 Broken Funnel Transitions:** Table showing the transition, affected session count, % of all drop-offs, and mean surprisal score.
  3. **Color-Coded Session Traces:** 10 sample abandoned sessions rendered as horizontal event pills—shaded green for low surprisal and **bright red** on the exact breakpoint event, with exact loss values on hover.

## 5. Verification & Acceptance Criteria

Before marking the project complete, the agent must verify:

1. `pytest` passes all unit tests (tokenizer round-trip, causal attention mask shape, forward pass logits shape, loss computation).
2. Running `python data/generate_sample.py` followed by `funnel-debug run data/sample_events.csv --goal purchase --epochs 5 --out report.html` completes without errors.
3. The generated `report.html` successfully surfaces the 3 planted UX bugs from the synthetic dataset in the **Top 5 Broken Funnel Transitions** table.
