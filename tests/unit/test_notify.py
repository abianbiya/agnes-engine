"""Checks for the Telegram end-of-conversation notifier."""

from datetime import datetime, timedelta
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.api.notify import find_ended_sessions, format_transcript


def _memory(**ages_in_minutes):
    now = datetime.now()
    return SimpleNamespace(
        last_accessed={
            sid: now - timedelta(minutes=age) for sid, age in ages_in_minutes.items()
        }
    )


def test_only_idle_sessions_are_reported():
    mem = _memory(fresh=1, stale=9)
    ended = find_ended_sessions(mem, {}, idle_minutes=5)
    assert [sid for sid, _ in ended] == ["stale"]


def test_does_not_notify_twice_for_the_same_idle_period():
    mem = _memory(stale=9)
    notified = {}
    for sid, last in find_ended_sessions(mem, notified, idle_minutes=5):
        notified[sid] = last

    assert find_ended_sessions(mem, notified, idle_minutes=5) == []


def test_notifies_again_after_the_user_comes_back():
    mem = _memory(stale=9)
    notified = {sid: last for sid, last in find_ended_sessions(mem, {}, idle_minutes=5)}

    # User sends another message, then goes idle again.
    mem.last_accessed["stale"] = datetime.now() - timedelta(minutes=6)

    assert [sid for sid, _ in find_ended_sessions(mem, notified, idle_minutes=5)] == ["stale"]


def test_forgets_evicted_sessions():
    mem = _memory(stale=9)
    notified = {"gone": datetime.now(), "stale": datetime.now()}
    find_ended_sessions(mem, notified, idle_minutes=5)
    assert "gone" not in notified


def test_transcript_fits_telegram_limit():
    messages = [HumanMessage(content="x" * 5000), AIMessage(content="y" * 5000)]
    text = format_transcript("abc", messages, datetime.now())

    assert len(text) <= 4096
    assert "User:" in text and "dipotong" in text


if __name__ == "__main__":
    test_only_idle_sessions_are_reported()
    test_does_not_notify_twice_for_the_same_idle_period()
    test_notifies_again_after_the_user_comes_back()
    test_forgets_evicted_sessions()
    test_transcript_fits_telegram_limit()
    print("ok")
