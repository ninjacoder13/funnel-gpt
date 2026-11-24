"""Download a real, zero-PII e-commerce clickstream slice from the REES46 dataset.

Source: REES46 Multi-Category Store eCommerce Behavior Data (Oct-Nov 2019),
hosted publicly on Hugging Face:
https://huggingface.co/datasets/kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019

Uses an HTTP Range request so only the requested prefix (e.g., 25 MB / ~200k events)
is streamed—no account, no API key, and no 14 GB full download required.
Outputs a 3-column zero-PII CSV: `session_id,timestamp,event_name`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
from pathlib import Path
import urllib.request

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019/"
    "resolve/main/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019.csv"
)


def _make_event_name(event_type: str, category_code: str, granularity: str) -> str:
    ev = event_type.strip().lower()
    if ev == "purchase" or granularity == "stage":
        return ev
    top_cat = category_code.strip().split(".")[0] if category_code.strip() else "uncategorized"
    return f"{ev}:{top_cat}"


def fetch_real_ecommerce_csv(
    output_path: str | Path = "data/real_ecommerce_events.csv",
    download_mb: int = 25,
    min_events_per_session: int = 2,
    granularity: str = "category",
) -> Path:
    """Stream a prefix of the REES46 clickstream CSV and convert to Funnel-GPT 3-column format."""
    byte_limit = download_mb * 1024 * 1024
    req = urllib.request.Request(
        DATASET_URL,
        headers={
            "Range": f"bytes=0-{byte_limit - 1}",
            "User-Agent": "funnel-gpt/0.1.0",
        },
    )

    print(f"[fetch_real_data] Streaming first {download_mb} MB from Hugging Face REES46 dataset...")
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_bytes = resp.read()

    # Decode and drop the final potentially truncated line from the byte-range slice
    text = raw_bytes.decode("utf-8", errors="ignore")
    last_nl = text.rfind("\n")
    if last_nl != -1:
        text = text[:last_nl]

    reader = csv.DictReader(io.StringIO(text))
    sessions: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for row in reader:
        sid = (row.get("user_session") or "").strip()
        ts = (row.get("event_time") or "").strip()
        ev_type = (row.get("event_type") or "").strip()
        cat = (row.get("category_code") or "").strip()
        if not sid or not ts or not ev_type:
            continue
        event_name = _make_event_name(ev_type, cat, granularity)
        sessions[sid].append((ts, event_name))

    # Filter to sessions with at least min_events_per_session events
    filtered = {
        sid: evs for sid, evs in sessions.items() if len(evs) >= min_events_per_session
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    event_counts: Counter[str] = Counter()
    converting_count = 0

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "timestamp", "event_name"])
        for sid, evs in filtered.items():
            if any(ev == "purchase" for _, ev in evs):
                converting_count += 1
            for ts, ev in evs:
                writer.writerow([sid, ts, ev])
                event_counts[ev] += 1

    total_sessions = len(filtered)
    total_events = sum(event_counts.values())
    print(
        f"[fetch_real_data] Wrote {total_sessions:,} real sessions "
        f"({converting_count:,} converting, {total_sessions - converting_count:,} abandoned; "
        f"{total_events:,} total events, {len(event_counts)} unique event tokens) -> {out}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a real e-commerce clickstream slice (REES46) for Funnel-GPT."
    )
    parser.add_argument(
        "--out",
        default="data/real_ecommerce_events.csv",
        help="Output 3-column CSV path",
    )
    parser.add_argument(
        "--mb",
        type=int,
        default=25,
        help="Megabytes to stream via HTTP Range request (default: 25)",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=2,
        help="Minimum events per session to include (default: 2)",
    )
    parser.add_argument(
        "--granularity",
        choices=["category", "stage"],
        default="category",
        help="Token granularity: 'category' (e.g. view:electronics, cart:electronics, purchase) or 'stage' (view, cart, purchase)",
    )
    args = parser.parse_args()

    fetch_real_ecommerce_csv(
        output_path=args.out,
        download_mb=args.mb,
        min_events_per_session=args.min_events,
        granularity=args.granularity,
    )


if __name__ == "__main__":
    main()
