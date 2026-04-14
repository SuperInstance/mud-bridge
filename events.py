"""
events.py — MUD Event System

Structured event types and parser for converting raw MUD text output
into typed, queryable events. Each event carries metadata (timestamp,
source, session) so consumers can filter and react precisely.

Event types cover the full lifecycle of a MUD session: room transitions,
chat messages, combat rounds, system notices, and error conditions.
"""

from __future__ import annotations

import enum
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque
import threading


# ---------------------------------------------------------------------------
# Event type enumeration
# ---------------------------------------------------------------------------

class EventType(enum.Enum):
    """Categorises every kind of output the MUD can produce."""
    ROOM_ENTER = "room_enter"       # Agent moved into a new room
    ROOM_DESC = "room_desc"         # Full room description displayed
    MESSAGE = "message"             # General / public message
    WHISPER = "whisper"             # Private agent-to-agent message
    SHOUT = "shout"                 # Room-wide shout
    PROMPT = "prompt"               # Interactive prompt (HP bar, etc.)
    SYSTEM = "system"               # Server / system notification
    COMBAT = "combat"               # Combat round result
    ERROR = "error"                 # Error condition
    INVENTORY = "inventory"         # Inventory listing
    LOOK = "look"                   # Look command output
    UNKNOWN = "unknown"             # Unparseable raw text


# ---------------------------------------------------------------------------
# Structured event
# ---------------------------------------------------------------------------

@dataclass
class MUDEvent:
    """A single structured MUD event.

    Attributes:
        event_type: The category of event.
        data: Free-form payload (varies by event_type).
        source: Origin of the event (agent name, "system", etc.).
        raw: The original raw text from the MUD.
        timestamp: Unix epoch when the event was created.
        event_id: Unique identifier for deduplication / tracing.
    """
    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    raw: str = ""
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "source": self.source,
            "raw": self.raw,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MUDEvent:
        """Deserialise from a dictionary."""
        return cls(
            event_type=EventType(d["event_type"]),
            data=d.get("data", {}),
            source=d.get("source", ""),
            raw=d.get("raw", ""),
            timestamp=d.get("timestamp", time.time()),
            event_id=d.get("event_id", uuid.uuid4().hex[:12]),
        )

    def __str__(self) -> str:
        return f"[{self.event_type.value}] {self.raw}"


# ---------------------------------------------------------------------------
# Event queue (thread-safe, bounded)
# ---------------------------------------------------------------------------

class EventQueue:
    """Per-session FIFO queue of MUDEvents.

    Thread-safe: producers (TCP reader) and consumers (HTTP /output) can
    operate concurrently.  The queue is bounded to prevent unbounded memory
    growth when consumers are slow.
    """

    def __init__(self, max_size: int = 1024) -> None:
        self._queue: deque[MUDEvent] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._new_event = threading.Event()
        self._max_size = max_size

    def push(self, event: MUDEvent) -> None:
        """Append an event.  Drops oldest if the queue is full."""
        with self._lock:
            self._queue.append(event)
        self._new_event.set()  # wake any long-polling waiter

    def push_many(self, events: list[MUDEvent]) -> None:
        """Append multiple events at once."""
        with self._lock:
            for e in events:
                self._queue.append(e)
        self._new_event.set()

    def drain(self, max_items: int = 50) -> list[MUDEvent]:
        """Remove and return up to *max_items* events."""
        with self._lock:
            items: list[MUDEvent] = []
            for _ in range(min(max_items, len(self._queue))):
                items.append(self._queue.popleft())
            return items

    def wait_for_events(self, timeout: float = 30.0) -> list[MUDEvent]:
        """Block until at least one event arrives or *timeout* expires.

        Returns all available events (up to 50).
        """
        self._new_event.wait(timeout=timeout)
        self._new_event.clear()
        return self.drain(max_items=50)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
        self._new_event.clear()

    def peek_all(self) -> list[MUDEvent]:
        """Return a snapshot without removing events (for diagnostics)."""
        with self._lock:
            return list(self._queue)


# ---------------------------------------------------------------------------
# MUD output parser
# ---------------------------------------------------------------------------

# Pre-compiled patterns — order matters: more specific first.

_PATTERNS: list[tuple[re.Pattern[str], EventType, dict[str, str]]] = [
    # Whisper:  [AgentName] whispers: <text>
    (
        re.compile(
            r"^\[(?P<source>[^\]]+)\]\s+whispers?:\s*(?P<text>.+)$", re.IGNORECASE
        ),
        EventType.WHISPER,
        {"text": "text"},
    ),
    # Shout:  <Name> shouts: <text>
    (
        re.compile(
            r"^(?P<source>[A-Za-z0-9_]+)\s+shouts?:\s*(?P<text>.+)$", re.IGNORECASE
        ),
        EventType.SHOUT,
        {"text": "text"},
    ),
    # Room title line:  --- Room Name ---
    (
        re.compile(r"^---\s*(?P<room_name>.+?)\s*---\s*$"),
        EventType.ROOM_ENTER,
        {"room_name": "room_name"},
    ),
    # Room description:  [Room] Description text
    (
        re.compile(r"^\[Room\]\s*(?P<description>.+)$"),
        EventType.ROOM_DESC,
        {"description": "description"},
    ),
    # System messages
    (
        re.compile(r"^\[System\]\s*(?P<text>.+)$", re.IGNORECASE),
        EventType.SYSTEM,
        {"text": "text"},
    ),
    # Combat messages
    (
        re.compile(
            r"^\[Combat\]\s*(?P<text>.+)$", re.IGNORECASE
        ),
        EventType.COMBAT,
        {"text": "text"},
    ),
    # Error messages
    (
        re.compile(
            r"^\[Error\]\s*(?P<text>.+)$", re.IGNORECASE
        ),
        EventType.ERROR,
        {"text": "text"},
    ),
    # Inventory listing
    (
        re.compile(r"^\[Inventory\]\s*(?P<text>.+)$", re.IGNORECASE),
        EventType.INVENTORY,
        {"text": "text"},
    ),
    # Prompt lines — typically short, ending with > or $
    (
        re.compile(r"^(?P<prompt>.{1,80}[>#]\s*)$"),
        EventType.PROMPT,
        {"prompt": "prompt"},
    ),
    # Public message:  <Name> says: <text>
    (
        re.compile(
            r"^(?P<source>[A-Za-z0-9_]+)\s+says?:\s*(?P<text>.+)$", re.IGNORECASE
        ),
        EventType.MESSAGE,
        {"text": "text"},
    ),
    # Look output
    (
        re.compile(r"^\[Look\]\s*(?P<text>.+)$", re.IGNORECASE),
        EventType.LOOK,
        {"text": "text"},
    ),
]


class MUDParser:
    """Converts raw MUD text lines into structured :class:`MUDEvent` objects.

    Usage::

        parser = MUDParser()
        events = parser.parse(raw_text)
    """

    def parse(self, raw: str) -> list[MUDEvent]:
        """Parse one or more lines of raw MUD output.

        Splits on ``\\n``, matches each line against known patterns,
        and returns a list of events.  Unmatched lines are emitted as
        ``UNKNOWN`` events so nothing is silently dropped.
        """
        events: list[MUDEvent] = []
        for line in raw.split("\n"):
            line = line.rstrip("\r")
            if not line.strip():
                continue
            events.append(self._parse_line(line))
        return events

    def _parse_line(self, line: str) -> MUDEvent:
        """Match a single line against all patterns (first match wins)."""
        for pattern, event_type, field_map in _PATTERNS:
            m = pattern.match(line)
            if m:
                data: dict[str, Any] = {}
                for key, group_name in field_map.items():
                    data[key] = m.group(group_name)
                return MUDEvent(
                    event_type=event_type,
                    data=data,
                    source=m.group("source") if "source" in m.groupdict() else "",
                    raw=line,
                )
        # Fallback: emit as UNKNOWN
        return MUDEvent(
            event_type=EventType.UNKNOWN,
            data={"text": line},
            raw=line,
        )

    # -- Convenience helpers for common event construction ------------------

    @staticmethod
    def make_system(text: str, source: str = "system") -> MUDEvent:
        return MUDEvent(
            event_type=EventType.SYSTEM,
            data={"text": text},
            source=source,
        )

    @staticmethod
    def make_error(text: str, source: str = "system") -> MUDEvent:
        return MUDEvent(
            event_type=EventType.ERROR,
            data={"text": text},
            source=source,
        )

    @staticmethod
    def make_message(text: str, source: str = "") -> MUDEvent:
        return MUDEvent(
            event_type=EventType.MESSAGE,
            data={"text": text},
            source=source,
        )
