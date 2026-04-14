"""
session.py — MUD Session Manager

Manages a single agent's TCP connection to the Holodeck MUD server.
Each session owns a socket, a read-buffer, and an EventQueue.  A background
thread continuously reads from the socket, parses output via MUDParser,
and enqueues structured events for the HTTP bridge to deliver.

The session lifecycle:
    1. ``connect()``  — open TCP socket, begin login handshake
    2. ``send()``     — write a command string to the MUD
    3. ``recv_events()`` — pull parsed events (used by /output long-poll)
    4. ``close()``    — tear down socket + reader thread
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any, Optional

from events import EventQueue, EventType, MUDEvent, MUDParser

logger = logging.getLogger("mud-bridge.session")

# How long to wait for the MUD to accept a connection.
_CONNECT_TIMEOUT_S: float = 10.0
# Idle timeout — if no data received in this window, assume dead.
_IDLE_TIMEOUT_S: float = 120.0
# Receive buffer size for socket.recv().
_RECV_BUF: int = 4096
# Encoding used by the Holodeck MUD.
_ENCODING: str = "utf-8"


class MUDSession:
    """Manages a single agent's MUD session via TCP.

    Attributes:
        session_id: Unique identifier assigned by the bridge.
        agent_name: Display name inside the MUD.
        agent_class: Character class (e.g. "engineer", "explorer").
        host: MUD server hostname or IP.
        port: MUD server TCP port.
        connected: Whether the TCP socket is live.
        event_queue: Thread-safe queue of parsed MUDEvents.
    """

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        agent_class: str,
        host: str = "127.0.0.1",
        port: int = 7777,
    ) -> None:
        self.session_id: str = session_id
        self.agent_name: str = agent_name
        self.agent_class: str = agent_class
        self.host: str = host
        self.port: int = port

        self._sock: Optional[socket.socket] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._buffer: str = ""
        self.connected: bool = False
        self.created_at: float = time.time()
        self.last_activity: float = self.created_at

        self.event_queue: EventQueue = EventQueue()
        self._parser: MUDParser = MUDParser()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a TCP connection to the MUD server and perform the login
        handshake.

        Raises:
            ConnectionError: If the MUD server is unreachable.
            OSError: On low-level socket failures.
        """
        if self.connected:
            logger.warning("[%s] already connected", self.session_id)
            return

        logger.info(
            "[%s] connecting to %s:%d as '%s' (%s)",
            self.session_id, self.host, self.port, self.agent_name, self.agent_class,
        )

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(_CONNECT_TIMEOUT_S)
        try:
            self._sock.connect((self.host, self.port))
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            self._sock.close()
            self._sock = None
            raise ConnectionError(
                f"Cannot reach MUD at {self.host}:{self.port} — {exc}"
            ) from exc

        self.connected = True
        self.last_activity = time.time()

        # Perform the login sequence.
        self._login()

        # Start the background reader thread.
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"session-{self.session_id}",
            daemon=True,
        )
        self._reader_thread.start()

        logger.info("[%s] connected ✓", self.session_id)

    def _login(self) -> None:
        """Send the character name and class to complete the MUD login.

        The Holodeck MUD expects:
            1. A welcome banner (which we read and discard / enqueue)
            2. The agent name
            3. The agent class
        """
        # Read the welcome banner — block briefly for it.
        welcome = self._read_raw(timeout=5.0)
        if welcome:
            events = self._parser.parse(welcome)
            self.event_queue.push_many(events)

        # Send credentials.
        self._send_raw(self.agent_name)
        time.sleep(0.1)
        self._send_raw(self.agent_class)
        time.sleep(0.2)

        # Read the post-login response.
        post_login = self._read_raw(timeout=5.0)
        if post_login:
            events = self._parser.parse(post_login)
            self.event_queue.push_many(events)

    def close(self) -> None:
        """Gracefully disconnect from the MUD."""
        logger.info("[%s] disconnecting", self.session_id)
        self._stop_event.set()
        self.connected = False

        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3.0)

        self.event_queue.push(
            MUDEvent(
                event_type=EventType.SYSTEM,
                data={"text": f"Session {self.session_id} closed."},
                source="bridge",
            )
        )
        logger.info("[%s] disconnected ✓", self.session_id)

    # ------------------------------------------------------------------
    # Sending commands
    # ------------------------------------------------------------------

    def send(self, command: str) -> None:
        """Send a command string to the MUD.

        Args:
            command: The raw command (e.g. "look", "go north").

        Raises:
            RuntimeError: If the session is not connected.
        """
        if not self.connected or not self._sock:
            raise RuntimeError(f"Session {self.session_id} is not connected")

        logger.debug("[%s] → %s", self.session_id, command)
        self._send_raw(command)
        self.last_activity = time.time()

    # ------------------------------------------------------------------
    # Receiving events (consumed by the bridge's /output endpoint)
    # ------------------------------------------------------------------

    def recv_events(self, timeout: float = 30.0) -> list[MUDEvent]:
        """Block until events are available or *timeout* expires.

        Returns:
            A list of :class:`MUDEvent` (may be empty on timeout).
        """
        if not self.connected:
            return [
                MUDEvent(
                    event_type=EventType.ERROR,
                    data={"text": "Session not connected"},
                    source="bridge",
                )
            ]
        events = self.event_queue.wait_for_events(timeout=timeout)
        if events:
            self.last_activity = time.time()
        return events

    def get_pending_events(self, max_items: int = 50) -> list[MUDEvent]:
        """Non-blocking drain of the event queue."""
        return self.event_queue.drain(max_items=max_items)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_raw(self, data: str) -> None:
        """Low-level write to the socket."""
        assert self._sock is not None  # for type-checkers
        try:
            self._sock.sendall((data + "\n").encode(_ENCODING))
        except OSError as exc:
            self.connected = False
            self.event_queue.push(
                MUDEvent(
                    event_type=EventType.ERROR,
                    data={"text": f"Send failed: {exc}"},
                    source="bridge",
                )
            )
            raise

    def _read_raw(self, timeout: float = 2.0) -> str:
        """Try to read available data within *timeout*.  Returns empty
        string if nothing arrived."""
        assert self._sock is not None
        self._sock.settimeout(timeout)
        try:
            data = self._sock.recv(_RECV_BUF)
            if not data:
                # EOF — MUD closed the connection.
                self.connected = False
                return ""
            return data.decode(_ENCODING, errors="replace")
        except socket.timeout:
            return ""
        except OSError as exc:
            logger.warning("[%s] read error: %s", self.session_id, exc)
            self.connected = False
            return ""

    def _reader_loop(self) -> None:
        """Background thread: continuously read from the socket, buffer
        incomplete lines, and push parsed events to the queue."""
        assert self._sock is not None
        self._sock.settimeout(2.0)  # short timeout for stop-event checks

        while not self._stop_event.is_set() and self.connected:
            try:
                data = self._sock.recv(_RECV_BUF)
                if not data:
                    logger.info("[%s] MUD closed connection", self.session_id)
                    break

                chunk = data.decode(_ENCODING, errors="replace")
                self._buffer += chunk
                self.last_activity = time.time()

                # Process complete lines.
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    events = self._parser.parse(line)
                    self.event_queue.push_many(events)

            except socket.timeout:
                # Normal — just loop back to check _stop_event.
                continue
            except OSError as exc:
                if not self._stop_event.is_set():
                    logger.warning("[%s] reader error: %s", self.session_id, exc)
                    self.event_queue.push(
                        MUDEvent(
                            event_type=EventType.ERROR,
                            data={"text": f"Connection lost: {exc}"},
                            source="bridge",
                        )
                    )
                break

        self.connected = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def is_idle(self, threshold: float = _IDLE_TIMEOUT_S) -> bool:
        """Return ``True`` if no data has flowed for *threshold* seconds."""
        return (time.time() - self.last_activity) > threshold

    def status_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly status summary."""
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "agent_class": self.agent_class,
            "host": self.host,
            "port": self.port,
            "connected": self.connected,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "pending_events": self.event_queue.size,
            "idle": self.is_idle(),
        }
