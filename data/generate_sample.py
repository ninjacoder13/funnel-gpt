"""Synthetic e-commerce funnel event generator for Funnel-GPT testing.

Generates a 3-column CSV (session_id, timestamp, event_name) with 5,000 sessions:
- 3,500 converting sessions ("happy paths" ending in `purchase`)
- 1,500 abandoned sessions containing 3 planted UX bugs:
    1. Payment error loop: `begin_checkout -> error_payment -> begin_checkout`
    2. Shipping bounce: `add_shipping -> page_view`
    3. 10-minute cart stall: `add_to_cart` -> (>=600s / DT_LONG) -> `begin_checkout`
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random


def _format_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_browse_prefix(rng: random.Random, start_time: datetime) -> tuple[list[tuple[datetime, str]], datetime]:
    """Generate realistic opening browsing events up to view_item."""
    events: list[tuple[datetime, str]] = [(start_time, "page_view")]
    t = start_time

    # Optional extra page_view or search
    if rng.random() < 0.35:
        t += timedelta(seconds=rng.randint(6, 45))
        events.append((t, "page_view"))

    # 1 to 3 view_item events
    n_views = rng.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
    for _ in range(n_views):
        t += timedelta(seconds=rng.randint(8, 55))
        events.append((t, "view_item"))

    return events, t


def generate_converting_session(rng: random.Random, start_time: datetime) -> list[tuple[datetime, str]]:
    """Generate a healthy converting session ending in `purchase`."""
    events, t = _generate_browse_prefix(rng, start_time)

    # add_to_cart (sometimes twice for multi-item baskets)
    t += timedelta(seconds=rng.randint(10, 120))
    events.append((t, "add_to_cart"))
    if rng.random() < 0.2:
        t += timedelta(seconds=rng.randint(10, 50))
        events.append((t, "view_item"))
        t += timedelta(seconds=rng.randint(10, 50))
        events.append((t, "add_to_cart"))

    # begin_checkout
    t += timedelta(seconds=rng.randint(5, 90))
    events.append((t, "begin_checkout"))

    # add_shipping
    t += timedelta(seconds=rng.randint(15, 180))
    events.append((t, "add_shipping"))

    # purchase
    t += timedelta(seconds=rng.randint(10, 120))
    events.append((t, "purchase"))
    return events


def generate_abandoned_session(
    rng: random.Random,
    start_time: datetime,
    bug_type: str,
) -> list[tuple[datetime, str]]:
    """Generate an abandoned session exhibiting one of the 3 planted UX bugs."""
    events, t = _generate_browse_prefix(rng, start_time)

    if bug_type == "payment_error_loop":
        t += timedelta(seconds=rng.randint(10, 120))
        events.append((t, "add_to_cart"))
        t += timedelta(seconds=rng.randint(5, 90))
        events.append((t, "begin_checkout"))
        # Planted Bug 1: stuck in begin_checkout -> error_payment -> begin_checkout loop
        loops = rng.choice([1, 2])
        for _ in range(loops):
            t += timedelta(seconds=rng.randint(1, 4))  # DT_INSTANT error
            events.append((t, "error_payment"))
            t += timedelta(seconds=rng.randint(5, 30))
            events.append((t, "begin_checkout"))
        return events

    if bug_type == "shipping_bounce":
        t += timedelta(seconds=rng.randint(10, 120))
        events.append((t, "add_to_cart"))
        t += timedelta(seconds=rng.randint(5, 90))
        events.append((t, "begin_checkout"))
        t += timedelta(seconds=rng.randint(15, 150))
        events.append((t, "add_shipping"))
        # Planted Bug 2: sudden add_shipping -> page_view bounce
        t += timedelta(seconds=rng.randint(1, 4))  # DT_INSTANT unexpected redirect
        events.append((t, "page_view"))
        return events

    if bug_type == "cart_stall":
        t += timedelta(seconds=rng.randint(10, 120))
        events.append((t, "add_to_cart"))
        # Planted Bug 3: 10+ minute stall (DT_LONG, >=600s) at add_to_cart
        t += timedelta(seconds=rng.randint(620, 1400))
        events.append((t, "begin_checkout"))
        return events

    raise ValueError(f"Unknown bug_type: {bug_type}")


def generate_dataset(
    output_path: str | Path = "data/sample_events.csv",
    n_converting: int = 3500,
    n_abandoned: int = 1500,
    seed: int = 42,
) -> Path:
    """Generate the synthetic dataset and write it to CSV."""
    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

    sessions: list[tuple[str, list[tuple[datetime, str]]]] = []

    for i in range(n_converting):
        start = base_time + timedelta(minutes=rng.randint(0, 60 * 24 * 30))
        sessions.append((f"conv_{i:05d}", generate_converting_session(rng, start)))

    bug_types = ["payment_error_loop", "shipping_bounce", "cart_stall"]
    for i in range(n_abandoned):
        start = base_time + timedelta(minutes=rng.randint(0, 60 * 24 * 30))
        bug_type = bug_types[i % len(bug_types)]
        sessions.append((f"drop_{i:05d}", generate_abandoned_session(rng, start, bug_type)))

    rng.shuffle(sessions)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "timestamp", "event_name"])
        for session_id, events in sessions:
            for ts, event_name in events:
                writer.writerow([session_id, _format_ts(ts), event_name])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Funnel-GPT e-commerce event CSV.")
    parser.add_argument("--out", default="data/sample_events.csv", help="Output CSV path")
    parser.add_argument("--converting", type=int, default=3500, help="Number of converting sessions")
    parser.add_argument("--abandoned", type=int, default=1500, help="Number of abandoned sessions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = generate_dataset(
        output_path=args.out,
        n_converting=args.converting,
        n_abandoned=args.abandoned,
        seed=args.seed,
    )
    print(f"Generated {args.converting + args.abandoned} sessions ({args.converting} converting, {args.abandoned} abandoned) -> {out}")


if __name__ == "__main__":
    main()
