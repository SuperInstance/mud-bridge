"""
bridge.py — MUD HTTP Bridge

An HTTP→MUD bridge server that allows standalone agents to interact with
the Holodeck MUD server programmatically via HTTP instead of raw telnet/TCP.

Architecture
------------
    Agent (HTTP client)
        │
        ▼
    ┌──────────────────────────────────────┐
    │           MudBridge (HTTP)            │
    │  /connect  /command  /output  …       │
    │                                      │
    │  ┌──────────┐ ┌──────────┐          │
    │  │ Session 1 │ │ Session 2 │  …      │
    │  └─────┬────┘ └─────┬────┘          │
    └────────┼────────────┼────────────────┘
             │            │        (TCP)
             ▼            ▼
        Holodeck MUD Server (:7777)

Endpoints
---------
    POST /connect     — create a new MUD session for an agent
    POST /command     — send a command on behalf of an agent
    GET  /output      — long-poll for pending MUD output
    POST /disconnect  — tear down an agent session
    GET  /rooms       — list available rooms (from connected agents)
    GET  /agents      — list all connected agents
    POST /whisper     — agent-to-agent private message
    GET  /status      — bridge health / diagnostics
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

from session import MUDSession
from events import MUDEvent, MUDParser

logger = logging.getLogger("mud-bridge")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BRIDGE_PORT: int = 8877
DEFAULT_MUD_HOST: str = "127.0.0.1"
DEFAULT_MUD_PORT: int = 7777
LONG_POLL_TIMEOUT_S: float = 30.0


# ---------------------------------------------------------------------------
# Session registry (thread-safe)
# ---------------------------------------------------------------------------

class SessionRegistry:
    """Thin thread-safe wrapper around the session map."""

    def __init__(self) -> None:
        self._sessions: dict[str, MUDSession] = {}
        self._lock = threading.Lock()

    def add(self, session: MUDSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Optional[MUDSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> Optional[MUDSession]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def find_by_name(self, agent_name: str) -> Optional[MUDSession]:
        with self._lock:
            for s in self._sessions.values():
                if s.agent_name.lower() == agent_name.lower():
                    return s
        return None

    def all_sessions(self) -> list[MUDSession]:
        with self._lock:
            return list(self._sessions.values())

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class BridgeHandler(BaseHTTPRequestHandler):
    """Routes HTTP requests to the :class:`MudBridge`."""

    # Class-level reference — set by MudBridge before serving.
    bridge: MudBridge = ...  # type: ignore[assignment]

    # Suppress default stderr logging for cleaner output.
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("HTTP %s", fmt % args)

    # -- helpers -------------------------------------------------------------

    def _read_json_body(self) -> dict[str, Any]:
        """Parse a JSON request body.  Returns empty dict on failure."""
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            raw = self.rfile.read(length)
            return json.loads(raw) if raw else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("bad JSON body: %s", exc)
            return {}

    def _send_json(self, status: int, obj: dict[str, Any]) -> None:
        """Write a JSON response."""
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _send_ok(self, data: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"ok": True}
        if data:
            payload.update(data)
        self._send_json(200, payload)

    def _get_param(
        self, params: dict[str, list[str]], key: str, default: str = ""
    ) -> str:
        values = params.get(key, [])
        return values[0] if values else default

    # -- routing -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path.rstrip("/")

        routes: dict[str, Any] = {
            "/output": self._handle_output,
            "/rooms": self._handle_rooms,
            "/agents": self._handle_agents,
            "/status": self._handle_status,
        }

        handler = routes.get(path)
        if handler:
            handler(params)
        else:
            self._send_error(404, f"Unknown route: GET {path}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        body = self._read_json_body()
        path = parsed.path.rstrip("/")

        routes: dict[str, Any] = {
            "/connect": self._handle_connect,
            "/command": self._handle_command,
            "/disconnect": self._handle_disconnect,
            "/whisper": self._handle_whisper,
        }

        handler = routes.get(path)
        if handler:
            handler(body, params=parse_qs(parsed.query))
        else:
            self._send_error(404, f"Unknown route: POST {path}")

    # -- endpoint handlers ---------------------------------------------------

    def _handle_connect(
        self, body: dict[str, Any], params: Any = None
    ) -> None:
        """POST /connect — create a new MUD session."""
        agent_name = body.get("agent_name", "").strip()
        agent_class = body.get("agent_class", "explorer").strip()

        if not agent_name:
            return self._send_error(400, "agent_name is required")

        # Check for duplicate name.
        existing = self.bridge.registry.find_by_name(agent_name)
        if existing and existing.connected:
            return self._send_error(
                409,
                f"Agent '{agent_name}' is already connected "
                f"(session: {existing.session_id})",
            )

        session_id = secrets.token_hex(8)
        session = MUDSession(
            session_id=session_id,
            agent_name=agent_name,
            agent_class=agent_class,
            host=self.bridge.mud_host,
            port=self.bridge.mud_port,
        )

        try:
            session.connect()
        except ConnectionError as exc:
            return self._send_error(502, str(exc))

        self.bridge.registry.add(session)

        # Collect any initial events (welcome banner, etc.)
        initial_events = session.get_pending_events()

        self._send_ok({
            "session_id": session.session_id,
            "agent_name": session.agent_name,
            "agent_class": session.agent_class,
            "connected": True,
            "initial_events": [e.to_dict() for e in initial_events],
        })

        logger.info(
            "Agent '%s' connected → session %s", agent_name, session_id
        )

    def _handle_command(
        self, body: dict[str, Any], params: Any = None
    ) -> None:
        """POST /command — send a command to the MUD."""
        session_id = body.get("session_id", "").strip()
        command = body.get("command", "").strip()

        if not session_id:
            return self._send_error(400, "session_id is required")
        if not command:
            return self._send_error(400, "command is required")

        session = self.bridge.registry.get(session_id)
        if not session:
            return self._send_error(404, f"Session '{session_id}' not found")
        if not session.connected:
            return self._send_error(410, f"Session '{session_id}' is disconnected")

        try:
            session.send(command)
        except RuntimeError as exc:
            return self._send_error(502, str(exc))

        # Brief drain — collect any immediate response (most MUDs respond
        # within a few hundred ms).  Remaining output goes to /output.
        time.sleep(0.15)
        immediate = session.get_pending_events(max_items=50)

        self._send_ok({
            "session_id": session_id,
            "command": command,
            "response": [e.to_dict() for e in immediate],
        })

    def _handle_output(self, params: dict[str, list[str]]) -> None:
        """GET /output — long-poll for pending MUD events."""
        session_id = self._get_param(params, "session_id")
        timeout_str = self._get_param(params, "timeout", str(int(LONG_POLL_TIMEOUT_S)))
        try:
            timeout = min(float(timeout_str), 60.0)
        except ValueError:
            timeout = LONG_POLL_TIMEOUT_S

        if not session_id:
            return self._send_error(400, "session_id query param is required")

        session = self.bridge.registry.get(session_id)
        if not session:
            return self._send_error(404, f"Session '{session_id}' not found")

        # Long-poll: block until events arrive or timeout.
        events = session.recv_events(timeout=timeout)

        self._send_ok({
            "session_id": session_id,
            "events": [e.to_dict() for e in events],
            "count": len(events),
            "connected": session.connected,
        })

    def _handle_disconnect(
        self, body: dict[str, Any], params: Any = None
    ) -> None:
        """POST /disconnect — close an agent's MUD session."""
        session_id = body.get("session_id", "").strip()

        if not session_id:
            return self._send_error(400, "session_id is required")

        session = self.bridge.registry.remove(session_id)
        if not session:
            return self._send_error(404, f"Session '{session_id}' not found")

        session.close()

        self._send_ok({
            "session_id": session_id,
            "disconnected": True,
        })
        logger.info("Session %s disconnected", session_id)

    def _handle_whisper(
        self, body: dict[str, Any], params: Any = None
    ) -> None:
        """POST /whisper — send a private message to another agent."""
        session_id = body.get("session_id", "").strip()
        target = body.get("target", "").strip()
        message = body.get("message", "").strip()

        if not session_id:
            return self._send_error(400, "session_id is required")
        if not target:
            return self._send_error(400, "target agent name is required")
        if not message:
            return self._send_error(400, "message is required")

        sender = self.bridge.registry.get(session_id)
        if not sender:
            return self._send_error(404, f"Session '{session_id}' not found")
        if not sender.connected:
            return self._send_error(410, "Sender session is disconnected")

        # The whisper is sent via the MUD's built-in whisper command.
        whisper_cmd = f"whisper {target} {message}"
        try:
            sender.send(whisper_cmd)
        except RuntimeError as exc:
            return self._send_error(502, str(exc))

        self._send_ok({
            "session_id": session_id,
            "target": target,
            "whisper": message,
        })

    def _handle_rooms(self, params: dict[str, list[str]]) -> None:
        """GET /rooms — list known rooms (best-effort from connected agents).

        The bridge asks each connected agent to issue a ``look`` command
        and reports the room names from the events.  For simplicity, we
        return rooms based on the session status data.
        """
        sessions = self.bridge.registry.all_sessions()
        rooms: list[dict[str, Any]] = []

        for s in sessions:
            if s.connected:
                rooms.append({
                    "agent": s.agent_name,
                    "session_id": s.session_id,
                    "status": "connected",
                    "pending_events": s.event_queue.size,
                })

        self._send_ok({"rooms": rooms, "count": len(rooms)})

    def _handle_agents(self, params: dict[str, list[str]]) -> None:
        """GET /agents — list all connected agents."""
        sessions = self.bridge.registry.all_sessions()
        agents = [s.status_dict() for s in sessions]
        self._send_ok({"agents": agents, "count": len(agents)})

    def _handle_status(self, params: dict[str, list[str]]) -> None:
        """GET /status — bridge health check and diagnostics."""
        uptime = time.time() - self.bridge.started_at
        sessions = self.bridge.registry.all_sessions()
        connected = sum(1 for s in sessions if s.connected)

        self._send_ok({
            "bridge": "mud-http-bridge",
            "version": "1.0.0",
            "uptime_s": round(uptime, 1),
            "mud_host": self.bridge.mud_host,
            "mud_port": self.bridge.mud_port,
            "total_sessions": self.bridge.registry.count,
            "connected_agents": connected,
            "idle_sessions": sum(1 for s in sessions if s.is_idle()),
        })


# ---------------------------------------------------------------------------
# Bridge server
# ---------------------------------------------------------------------------

class MudBridge:
    """HTTP→MUD bridge. Agents connect via HTTP, bridge manages MUD TCP connections.

    Usage::

        bridge = MudBridge(port=8877, mud_host="127.0.0.1", mud_port=7777)
        bridge.start()       # blocks
    """

    def __init__(
        self,
        port: int = DEFAULT_BRIDGE_PORT,
        mud_host: str = DEFAULT_MUD_HOST,
        mud_port: int = DEFAULT_MUD_PORT,
    ) -> None:
        self.port: int = port
        self.mud_host: str = mud_host
        self.mud_port: int = mud_port
        self.started_at: float = time.time()
        self.registry: SessionRegistry = SessionRegistry()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- public API ----------------------------------------------------------

    def start(self, blocking: bool = True) -> None:
        """Start the HTTP bridge server.

        Args:
            blocking: If True, blocks the calling thread.  If False, starts
                      a daemon thread and returns immediately.
        """
        BridgeHandler.bridge = self  # type: ignore[attr-defined]

        self._server = HTTPServer(("0.0.0.0", self.port), BridgeHandler)
        self._server.daemon_threads = True
        self.started_at = time.time()

        logger.info(
            "MUD HTTP Bridge listening on :%d → MUD at %s:%d",
            self.port, self.mud_host, self.mud_port,
        )

        if blocking:
            self._serve_forever()
        else:
            self._thread = threading.Thread(
                target=self._serve_forever,
                name="mud-bridge",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Shut down the bridge and close all agent sessions."""
        logger.info("Shutting down MUD bridge…")
        for session in self.registry.all_sessions():
            session.close()
        if self._server:
            self._server.shutdown()
        logger.info("MUD bridge stopped.")

    def start_background(self) -> None:
        """Convenience alias for ``start(blocking=False)``."""
        self.start(blocking=False)

    def wait(self) -> None:
        """Block until the background server thread exits."""
        if self._thread:
            self._thread.join()

    # -- internal ------------------------------------------------------------

    def _serve_forever(self) -> None:
        assert self._server is not None
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Received interrupt — stopping.")
        finally:
            self.stop()
