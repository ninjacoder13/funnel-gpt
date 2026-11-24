"""Unit tests for EventTokenizer and CSV session loading."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tokenizer import (
    DT_INSTANT,
    DT_LONG,
    DT_MIN,
    DT_SEC,
    DT_START,
    END_TOKEN,
    START_TOKEN,
    UNK_TOKEN,
    EventTokenizer,
    bucket_time_delta,
    load_sessions_from_csv,
)


def test_bucket_time_delta_boundaries() -> None:
    assert bucket_time_delta(None) == DT_START
    assert bucket_time_delta(0.0) == DT_INSTANT
    assert bucket_time_delta(4.99) == DT_INSTANT
    assert bucket_time_delta(5.0) == DT_SEC
    assert bucket_time_delta(59.99) == DT_SEC
    assert bucket_time_delta(60.0) == DT_MIN
    assert bucket_time_delta(599.99) == DT_MIN
    assert bucket_time_delta(600.0) == DT_LONG
    assert bucket_time_delta(3600.0) == DT_LONG


def test_tokenizer_round_trip_and_sorting() -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Deliberately out of order to verify chronological sorting
    session = [
        (t0 + timedelta(seconds=120), "add_to_cart"),
        (t0, "page_view"),
        (t0 + timedelta(seconds=720), "purchase"),
        (t0 + timedelta(seconds=20), "view_item"),
    ]
    tok = EventTokenizer().fit([session])
    ids = tok.encode_session(session)
    decoded = tok.decode(ids)

    assert decoded == [
        START_TOKEN,
        DT_START,
        "page_view",
        DT_SEC,
        "view_item",
        DT_MIN,
        "add_to_cart",
        DT_LONG,
        "purchase",
        END_TOKEN,
    ]


def test_tokenizer_unknown_event_maps_to_unk() -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    train_session = [(t0, "page_view"), (t0 + timedelta(seconds=10), "purchase")]
    tok = EventTokenizer().fit([train_session])

    unseen_session = [(t0, "page_view"), (t0 + timedelta(seconds=2), "error_payment")]
    decoded = tok.decode(tok.encode_session(unseen_session))
    assert decoded == [START_TOKEN, DT_START, "page_view", DT_INSTANT, UNK_TOKEN, END_TOKEN]


def test_tokenizer_save_and_load_json(tmp_path: Path) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    session = [(t0, "page_view"), (t0 + timedelta(seconds=15), "purchase")]
    tok = EventTokenizer().fit([session])

    save_path = tmp_path / "tokenizer.json"
    tok.save(save_path)
    loaded = EventTokenizer.load(save_path)

    assert loaded.token_to_id == tok.token_to_id
    assert loaded.encode_session(session) == tok.encode_session(session)


def test_load_sessions_from_csv(tmp_path: Path) -> None:
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(
        "session_id,timestamp,event_name\n"
        "s1,2026-01-01T10:01:00Z,view_item\n"
        "s1,2026-01-01T10:00:00Z,page_view\n"
        "s2,2026-01-01T11:00:00Z,page_view\n",
        encoding="utf-8",
    )
    sessions = load_sessions_from_csv(csv_file)
    assert set(sessions.keys()) == {"s1", "s2"}
    assert [ev for _, ev in sessions["s1"]] == ["page_view", "view_item"]
