"""Event + Time-Delta Discrete Tokenizer for Funnel-GPT."""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
START_TOKEN = "[START]"
END_TOKEN = "[END]"

SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN)

DT_START = "DT_START"
DT_INSTANT = "DT_INSTANT"
DT_SEC = "DT_SEC"
DT_MIN = "DT_MIN"
DT_LONG = "DT_LONG"

DT_TOKENS: tuple[str, ...] = (DT_START, DT_INSTANT, DT_SEC, DT_MIN, DT_LONG)


def parse_timestamp(raw: str) -> datetime:
    """Parse an ISO-8601 or common datetime string into a timezone-aware datetime."""
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Fallback for numeric unix timestamps (seconds or milliseconds)
        val = float(text)
        if val > 1e12:
            val /= 1000.0
        dt = datetime.fromtimestamp(val, tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def bucket_time_delta(dt_seconds: float | None) -> str:
    """Map elapsed seconds since previous event to a discrete time-delta token."""
    if dt_seconds is None:
        return DT_START
    if dt_seconds < 5.0:
        return DT_INSTANT
    if dt_seconds < 60.0:
        return DT_SEC
    if dt_seconds < 600.0:
        return DT_MIN
    return DT_LONG


def load_sessions_from_csv(csv_path: str | Path) -> dict[str, list[tuple[datetime, str]]]:
    """Read a 3-column CSV (session_id, timestamp, event_name) into chronologically sorted sessions."""
    sessions: dict[str, list[tuple[datetime, int, str]]] = defaultdict(list)
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"session_id", "timestamp", "event_name"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}, got {reader.fieldnames}"
            )
        for row_idx, row in enumerate(reader):
            sid = row["session_id"].strip()
            event_name = row["event_name"].strip()
            ts = parse_timestamp(row["timestamp"])
            sessions[sid].append((ts, row_idx, event_name))

    sorted_sessions: dict[str, list[tuple[datetime, str]]] = {}
    for sid, events in sessions.items():
        # Stable chronological sort (row_idx preserves file order on identical timestamps)
        events.sort(key=lambda item: (item[0], item[1]))
        sorted_sessions[sid] = [(ts, ev) for ts, _, ev in events]
    return sorted_sessions


def session_to_dt_events(
    events: Sequence[tuple[datetime, str]],
) -> list[tuple[str, str]]:
    """Sort events chronologically and pair each event with its DT bucket token."""
    ordered = sorted(enumerate(events), key=lambda item: (item[1][0], item[0]))
    pairs: list[tuple[str, str]] = []
    prev_ts: datetime | None = None
    for _, (ts, event_name) in ordered:
        dt_seconds = None if prev_ts is None else max(0.0, (ts - prev_ts).total_seconds())
        pairs.append((bucket_time_delta(dt_seconds), event_name))
        prev_ts = ts
    return pairs


class EventTokenizer:
    """Discrete tokenizer encoding each event as [DT_BUCKET, event_name] wrapped in [START]...[END]."""

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        if token_to_id is not None:
            self.token_to_id = dict(token_to_id)
        else:
            base_tokens = list(SPECIAL_TOKENS) + list(DT_TOKENS)
            self.token_to_id = {tok: idx for idx, tok in enumerate(base_tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def start_id(self) -> int:
        return self.token_to_id[START_TOKEN]

    @property
    def end_id(self) -> int:
        return self.token_to_id[END_TOKEN]

    def fit(self, sessions: Iterable[Sequence[tuple[datetime, str]]]) -> "EventTokenizer":
        """Build vocabulary from an iterable of sessions (each a sequence of (timestamp, event_name))."""
        event_names: set[str] = set()
        for events in sessions:
            for _, event_name in events:
                event_names.add(event_name)
        for name in sorted(event_names):
            if name not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[name] = idx
                self.id_to_token[idx] = name
        return self

    def tokenize_session(self, events: Sequence[tuple[datetime, str]]) -> list[str]:
        """Convert a session's (timestamp, event_name) list into string tokens."""
        tokens = [START_TOKEN]
        for dt_bucket, event_name in session_to_dt_events(events):
            tokens.append(dt_bucket)
            tokens.append(event_name)
        tokens.append(END_TOKEN)
        return tokens

    def encode_tokens(self, tokens: Sequence[str]) -> list[int]:
        """Map string tokens to integer IDs (unseen tokens map to [UNK])."""
        unk = self.unk_id
        return [self.token_to_id.get(tok, unk) for tok in tokens]

    def encode_session(self, events: Sequence[tuple[datetime, str]]) -> list[int]:
        """Encode a session directly into token IDs."""
        return self.encode_tokens(self.tokenize_session(events))

    def decode(self, token_ids: Sequence[int]) -> list[str]:
        """Decode integer token IDs back into string tokens."""
        return [self.id_to_token.get(int(idx), UNK_TOKEN) for idx in token_ids]

    def save(self, path: str | Path) -> Path:
        """Persist tokenizer vocabulary to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token_to_id": self.token_to_id}
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "EventTokenizer":
        """Load tokenizer vocabulary from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token_to_id=data["token_to_id"])
