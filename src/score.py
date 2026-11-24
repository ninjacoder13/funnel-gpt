"""Surprisal scoring and UX breakpoint clustering for Funnel-GPT."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.model import GPTModel
from src.tokenizer import EventTokenizer, START_TOKEN, session_to_dt_events


@dataclass
class ScoredEvent:
    index: int
    dt_bucket: str
    event_name: str
    surprisal: float
    is_breakpoint: bool = False


@dataclass
class SessionTrace:
    session_id: str
    events: list[ScoredEvent]
    breakpoint_index: int
    prev_event: str
    dt_bucket: str
    breakpoint_event: str
    breakpoint_surprisal: float
    exceeded_threshold: bool

    @property
    def transition_label(self) -> str:
        return f"{self.prev_event} -> {self.dt_bucket} -> {self.breakpoint_event}"


@dataclass
class BreakpointCluster:
    prev_event: str
    dt_bucket: str
    breakpoint_event: str
    count: int
    pct_dropoffs: float
    mean_surprisal: float

    @property
    def transition_label(self) -> str:
        return f"{self.prev_event} -> {self.dt_bucket} -> {self.breakpoint_event}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition": self.transition_label,
            "prev_event": self.prev_event,
            "dt_bucket": self.dt_bucket,
            "breakpoint_event": self.breakpoint_event,
            "count": self.count,
            "pct_dropoffs": round(self.pct_dropoffs, 2),
            "mean_surprisal": round(self.mean_surprisal, 4),
        }


def compute_token_losses_for_sequence(
    model: GPTModel,
    token_ids: Sequence[int],
    context_length: int,
    device: torch.device,
) -> list[float]:
    """Compute next-token cross-entropy loss for every target position in a single sequence."""
    if len(token_ids) < 2:
        return []

    n_targets = len(token_ids) - 1
    if n_targets <= context_length:
        inp = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)
        logits = model(inp)
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt.view(-1),
            reduction="none",
        )
        return losses.detach().cpu().tolist()

    # Long sequence: score first context_length targets, then slide for remaining targets
    all_losses: list[float] = []
    inp = torch.tensor([token_ids[:context_length]], dtype=torch.long, device=device)
    tgt = torch.tensor([token_ids[1 : context_length + 1]], dtype=torch.long, device=device)
    logits = model(inp)
    first_losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tgt.view(-1),
        reduction="none",
    )
    all_losses.extend(first_losses.detach().cpu().tolist())

    # Score remaining positions with full context window ending at each position
    for pos in range(context_length, n_targets):
        window_start = pos - context_length + 1
        inp_win = torch.tensor(
            [token_ids[window_start : pos + 1]], dtype=torch.long, device=device
        )
        tgt_id = torch.tensor([token_ids[pos + 1]], dtype=torch.long, device=device)
        logits_win = model(inp_win)[:, -1, :]
        loss_val = F.cross_entropy(logits_win, tgt_id).item()
        all_losses.append(loss_val)
    return all_losses


@torch.no_grad()
def compute_event_surprisals_batch(
    model: GPTModel,
    encoded_sessions: Sequence[Sequence[int]],
    pad_id: int,
    context_length: int,
    device: torch.device,
    batch_size: int = 128,
) -> list[list[float]]:
    """Compute per-event surprisal (mean CE loss of [DT_BUCKET, event_name]) for each session."""
    model.eval()
    results: list[list[float]] = [[] for _ in range(len(encoded_sessions))]

    short_indices: list[int] = []
    long_indices: list[int] = []
    for idx, seq in enumerate(encoded_sessions):
        if len(seq) < 4:  # Need at least [START], DT, event, [END]
            continue
        if len(seq) - 1 <= context_length:
            short_indices.append(idx)
        else:
            long_indices.append(idx)

    # Fast batched path for sessions fitting within context_length
    for start in range(0, len(short_indices), batch_size):
        chunk_idxs = short_indices[start : start + batch_size]
        seqs = [encoded_sessions[i] for i in chunk_idxs]
        max_len = max(len(s) - 1 for s in seqs)

        inputs = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        targets = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        for row_i, s in enumerate(seqs):
            length = len(s) - 1
            inputs[row_i, :length] = torch.tensor(s[:-1], dtype=torch.long, device=device)
            targets[row_i, :length] = torch.tensor(s[1:], dtype=torch.long, device=device)

        logits = model(inputs)
        token_losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view(len(seqs), max_len)
        token_losses_cpu = token_losses.detach().cpu().numpy()

        for row_i, orig_idx in enumerate(chunk_idxs):
            num_events = (len(encoded_sessions[orig_idx]) - 2) // 2
            ev_losses: list[float] = []
            for e_idx in range(num_events):
                dt_loss = float(token_losses_cpu[row_i, 2 * e_idx])
                ev_loss = float(token_losses_cpu[row_i, 2 * e_idx + 1])
                ev_losses.append(0.5 * (dt_loss + ev_loss))
            results[orig_idx] = ev_losses

    # Fallback sliding-window path for long sessions
    for orig_idx in long_indices:
        seq = encoded_sessions[orig_idx]
        tok_losses = compute_token_losses_for_sequence(model, seq, context_length, device)
        num_events = (len(seq) - 2) // 2
        ev_losses = [
            0.5 * (tok_losses[2 * e_idx] + tok_losses[2 * e_idx + 1])
            for e_idx in range(num_events)
        ]
        results[orig_idx] = ev_losses

    return results


def score_non_converting_sessions(
    model: GPTModel,
    tokenizer: EventTokenizer,
    non_converting_sessions: dict[str, list[tuple[datetime, str]]],
    p95_threshold: float,
    device: torch.device,
    top_k: int = 5,
    sample_traces_count: int = 10,
) -> dict[str, Any]:
    """Score all non-converting sessions, find breakpoints, and cluster Top-K transitions."""
    session_ids = list(non_converting_sessions.keys())
    raw_pairs_list = [session_to_dt_events(non_converting_sessions[sid]) for sid in session_ids]
    encoded_list = [tokenizer.encode_session(non_converting_sessions[sid]) for sid in session_ids]

    surprisals_per_session = compute_event_surprisals_batch(
        model=model,
        encoded_sessions=encoded_list,
        pad_id=tokenizer.pad_id,
        context_length=model.cfg["context_length"],
        device=device,
    )

    traces: list[SessionTrace] = []
    cluster_surprisals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    cluster_traces: dict[tuple[str, str, str], list[SessionTrace]] = defaultdict(list)

    for sid, pairs, ev_surprisals in zip(session_ids, raw_pairs_list, surprisals_per_session):
        if not pairs or not ev_surprisals:
            continue

        # Breakpoint: first event exceeding p95_threshold, else max-surprisal event
        exceeded = False
        bp_idx = 0
        for idx, val in enumerate(ev_surprisals):
            if val > p95_threshold:
                bp_idx = idx
                exceeded = True
                break
        if not exceeded:
            bp_idx = int(np.argmax(ev_surprisals))

        scored_events = [
            ScoredEvent(
                index=idx,
                dt_bucket=dt_b,
                event_name=ev_name,
                surprisal=float(s_val),
                is_breakpoint=(idx == bp_idx),
            )
            for idx, ((dt_b, ev_name), s_val) in enumerate(zip(pairs, ev_surprisals))
        ]

        prev_event = pairs[bp_idx - 1][1] if bp_idx > 0 else START_TOKEN
        dt_bucket, bp_event = pairs[bp_idx]
        bp_surprisal = float(ev_surprisals[bp_idx])

        trace = SessionTrace(
            session_id=sid,
            events=scored_events,
            breakpoint_index=bp_idx,
            prev_event=prev_event,
            dt_bucket=dt_bucket,
            breakpoint_event=bp_event,
            breakpoint_surprisal=bp_surprisal,
            exceeded_threshold=exceeded,
        )
        traces.append(trace)

        key = (prev_event, dt_bucket, bp_event)
        cluster_surprisals[key].append(bp_surprisal)
        cluster_traces[key].append(trace)

    total_scored = len(traces)
    clusters: list[BreakpointCluster] = []
    for (prev_ev, dt_b, bp_ev), vals in cluster_surprisals.items():
        count = len(vals)
        pct = (100.0 * count / total_scored) if total_scored > 0 else 0.0
        clusters.append(
            BreakpointCluster(
                prev_event=prev_ev,
                dt_bucket=dt_b,
                breakpoint_event=bp_ev,
                count=count,
                pct_dropoffs=pct,
                mean_surprisal=float(np.mean(vals)),
            )
        )

    # Rank by frequency descending, tie-breaking by mean surprisal descending
    clusters.sort(key=lambda c: (-c.count, -c.mean_surprisal))
    top_clusters = clusters[:top_k]

    # Select diverse sample traces round-robin across top clusters (up to sample_traces_count)
    sample_traces: list[SessionTrace] = []
    seen_sids: set[str] = set()
    if top_clusters:
        cluster_queues = [
            sorted(
                cluster_traces[(c.prev_event, c.dt_bucket, c.breakpoint_event)],
                key=lambda t: -t.breakpoint_surprisal,
            )
            for c in top_clusters
        ]
        round_idx = 0
        while len(sample_traces) < sample_traces_count:
            added_any = False
            for q in cluster_queues:
                if round_idx < len(q) and len(sample_traces) < sample_traces_count:
                    cand = q[round_idx]
                    if cand.session_id not in seen_sids:
                        sample_traces.append(cand)
                        seen_sids.add(cand.session_id)
                        added_any = True
            if not added_any:
                break
            round_idx += 1

    # Fill remaining slots if fewer than sample_traces_count
    if len(sample_traces) < sample_traces_count:
        for cand in sorted(traces, key=lambda t: -t.breakpoint_surprisal):
            if cand.session_id not in seen_sids:
                sample_traces.append(cand)
                seen_sids.add(cand.session_id)
            if len(sample_traces) >= sample_traces_count:
                break

    return {
        "total_non_converting_scored": total_scored,
        "p95_threshold": p95_threshold,
        "top_transitions": top_clusters,
        "sample_traces": sample_traces,
        "all_traces_count": total_scored,
    }
