"""Happy-path pretraining loop, validation P95 threshold, and checkpointing."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import create_dataloader
from src.model import GPTModel, MINI_GPT_CONFIG
from src.score import compute_event_surprisals_batch
from src.tokenizer import EventTokenizer, load_sessions_from_csv


def select_device() -> torch.device:
    """Auto-detect Apple Silicon MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_sessions_by_goal(
    sessions: dict[str, list[tuple[datetime, str]]],
    goal_event: str,
) -> tuple[dict[str, list[tuple[datetime, str]]], dict[str, list[tuple[datetime, str]]]]:
    """Partition sessions into converting (containing goal_event) and non-converting."""
    converting: dict[str, list[tuple[datetime, str]]] = {}
    non_converting: dict[str, list[tuple[datetime, str]]] = {}
    for sid, events in sessions.items():
        if any(ev == goal_event for _, ev in events):
            converting[sid] = events
        else:
            non_converting[sid] = events
    return converting, non_converting


def _evaluate_loss(
    model: GPTModel,
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_id,
            )
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(1, total_batches)


def train_funnel_gpt(
    csv_path: str | Path,
    goal_event: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    artifacts_dir: str | Path = "artifacts",
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train Funnel-GPT on converting sessions and persist model, tokenizer, and metrics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    resolved_device = device if device is not None else select_device()

    all_sessions = load_sessions_from_csv(csv_path)
    converting, non_converting = split_sessions_by_goal(all_sessions, goal_event)

    if len(converting) < 2:
        raise ValueError(
            f"Need at least 2 converting sessions containing goal '{goal_event}', found {len(converting)}."
        )

    # Fit tokenizer exclusively on converting sessions ("happy paths")
    tokenizer = EventTokenizer().fit(converting.values())

    # Deterministic 90/10 train/val split of converting sessions
    conv_sids = sorted(converting.keys())
    rng = random.Random(seed)
    rng.shuffle(conv_sids)
    val_size = max(1, int(round(0.1 * len(conv_sids))))
    val_sids = conv_sids[:val_size]
    train_sids = conv_sids[val_size:]

    train_encoded = [tokenizer.encode_session(converting[sid]) for sid in train_sids]
    val_encoded = [tokenizer.encode_session(converting[sid]) for sid in val_sids]

    cfg = dict(MINI_GPT_CONFIG)
    cfg["vocab_size"] = tokenizer.vocab_size

    train_loader = create_dataloader(
        encoded_sessions=train_encoded,
        context_length=cfg["context_length"],
        pad_id=tokenizer.pad_id,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        encoded_sessions=val_encoded,
        context_length=cfg["context_length"],
        pad_id=tokenizer.pad_id,
        batch_size=batch_size,
        shuffle=False,
    )

    model = GPTModel(cfg).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_history: list[dict[str, float]] = []
    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(resolved_device)
            targets = targets.to(resolved_device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_id,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        final_train_loss = running_loss / max(1, n_batches)
        final_val_loss = _evaluate_loss(model, val_loader, tokenizer.pad_id, resolved_device)
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": round(final_train_loss, 5),
                "val_loss": round(final_val_loss, 5),
            }
        )

    # Compute 95th percentile per-event surprisal threshold on validation converting sessions
    val_event_surprisals = compute_event_surprisals_batch(
        model=model,
        encoded_sessions=val_encoded,
        pad_id=tokenizer.pad_id,
        context_length=cfg["context_length"],
        device=resolved_device,
    )
    flat_val_surprisals = [s for sess in val_event_surprisals for s in sess]
    p95_threshold = float(np.percentile(flat_val_surprisals, 95)) if flat_val_surprisals else 0.0

    total_sessions = len(all_sessions)
    conversion_rate = (100.0 * len(converting) / total_sessions) if total_sessions > 0 else 0.0

    metrics: dict[str, Any] = {
        "goal_event": goal_event,
        "device": str(resolved_device),
        "total_sessions": total_sessions,
        "converting_sessions": len(converting),
        "non_converting_sessions": len(non_converting),
        "conversion_rate_pct": round(conversion_rate, 2),
        "vocab_size": tokenizer.vocab_size,
        "param_count": model.count_parameters(),
        "epochs": epochs,
        "final_train_loss": round(final_train_loss, 5),
        "final_val_loss": round(final_val_loss, 5),
        "p95_threshold": round(p95_threshold, 5),
        "epoch_history": epoch_history,
    }

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(out_dir / "tokenizer.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "p95_threshold": p95_threshold,
            "goal_event": goal_event,
        },
        out_dir / "model.pt",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "p95_threshold": p95_threshold,
        "device": resolved_device,
        "non_converting_sessions": non_converting,
        "artifacts_dir": out_dir,
    }
