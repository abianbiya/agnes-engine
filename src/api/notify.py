"""
Telegram notification for ended conversations.

A conversation is "ended" when its session has been idle for
TELEGRAM_IDLE_MINUTES. A background task started in the app lifespan polls the
in-process ConversationMemory and posts the full transcript once per idle
period.

Disabled unless TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import httpx
from dotenv import load_dotenv

from src.utils.logging import get_logger

logger = get_logger(__name__)

# pydantic-settings reads .env into its own object and never touches
# os.environ, so os.getenv() is blind to it outside Docker. One line beats
# adding a settings class for three optional strings.
load_dotenv()

# ponytail: plain env vars, not a pydantic settings block. Three optional
# strings that are read once at startup do not need a schema.
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
IDLE_MINUTES = float(os.getenv("TELEGRAM_IDLE_MINUTES", "5"))

POLL_SECONDS = 60
MAX_CHARS = 3900  # Telegram caps a message at 4096


def enabled() -> bool:
    """True when both token and chat id are configured."""
    return bool(BOT_TOKEN and CHAT_ID)


def format_transcript(session_id: str, messages: list, ended_at: datetime) -> str:
    """
    Render a session as a plain-text Telegram message.

    Args:
        session_id: Session identifier.
        messages: BaseMessage list from ConversationMemory.
        ended_at: Timestamp of the last activity.

    Returns:
        Message body, truncated to Telegram's limit.
    """
    header = (
        f"Percakapan Agnes selesai\n"
        f"Sesi: {session_id}\n"
        f"Berakhir: {ended_at:%Y-%m-%d %H:%M}\n"
        f"Pesan: {len(messages)}\n"
    )

    lines = []
    for msg in messages:
        who = "User" if msg.type == "human" else "Agnes"
        lines.append(f"\n{who}: {msg.content}")

    body = "".join(lines)
    room = MAX_CHARS - len(header)
    if len(body) > room:
        body = body[:room - 20] + "\n… (dipotong)"

    return header + body


def find_ended_sessions(
    memory,
    notified: Dict[str, datetime],
    now: datetime | None = None,
    idle_minutes: float | None = None,
) -> List[Tuple[str, datetime]]:
    """
    Find sessions idle long enough to count as ended and not yet reported.

    Re-notifies if the user came back and went idle again, because the
    timestamp recorded in `notified` no longer matches.

    Args:
        memory: ConversationMemory instance.
        notified: session_id -> last_accessed value already reported.
        now: Current time (injectable for tests).
        idle_minutes: Idle threshold (defaults to IDLE_MINUTES).

    Returns:
        List of (session_id, last_accessed) pairs to notify.
    """
    now = now or datetime.now()
    cutoff = timedelta(minutes=IDLE_MINUTES if idle_minutes is None else idle_minutes)

    snapshot = list(memory.last_accessed.items())

    # Drop bookkeeping for sessions memory has already evicted.
    alive = {sid for sid, _ in snapshot}
    for sid in [s for s in notified if s not in alive]:
        del notified[sid]

    return [
        (sid, last)
        for sid, last in snapshot
        if now - last >= cutoff and notified.get(sid) != last
    ]


async def send(text: str) -> None:
    """Post a message to Telegram. Never raises."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
            )
            r.raise_for_status()
    except Exception as e:  # pragma: no cover - notification must never break the API
        logger.warning("telegram_send_failed", error=str(e))


async def watch_sessions() -> None:
    """Poll conversation memory and notify on ended conversations."""
    if not enabled():
        logger.warning(
            "telegram_notify_disabled",
            has_token=bool(BOT_TOKEN),
            has_chat_id=bool(CHAT_ID),
        )
        return

    from src.api.dependencies import get_conversation_memory

    memory = get_conversation_memory()
    notified: Dict[str, datetime] = {}

    logger.info("telegram_notify_started", idle_minutes=IDLE_MINUTES)

    while True:
        await asyncio.sleep(POLL_SECONDS)
        try:
            for session_id, last in find_ended_sessions(memory, notified):
                # Direct read: get_messages() would bump last_accessed and the
                # session would look active again on every poll.
                history = memory.sessions.get(session_id)
                messages = history.messages if history else []
                if messages:
                    await send(format_transcript(session_id, messages, last))
                    logger.info("telegram_notified", session_id=session_id)
                notified[session_id] = last
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover
            logger.warning("telegram_watch_error", error=str(e))


__all__ = ["enabled", "format_transcript", "find_ended_sessions", "send", "watch_sessions"]
