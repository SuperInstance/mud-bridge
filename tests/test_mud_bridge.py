"""
tests/test_mud_bridge.py — Tests for the MUD HTTP Bridge

Comprehensive test suite covering:
- Event system (creation, serialisation, parsing, queue)
- Session management (connect, send, receive, close)
- Bridge HTTP endpoints (connect, command, output, disconnect, agents, status)
- Graceful disconnect and cleanup

Unit tests mock TCP via unittest.mock; integration tests use a lightweight
fake MUD TCP server so no real MUD is needed.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from typing import Any, Optional
from unittest.mock import MagicMock, patch, call
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import sys
import os

# Ensure the parent directory is on sys.path so we can import our modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from events import (
    EventType,
    MUDEvent,
    EventQueue,
    MUDParser,
)
from session import MUDSession
from bridge import MudBridge, BridgeHandler, SessionRegistry


# ===================================================================
# Helper: Fake MUD TCP Server for integration tests
# ===================================================================

class FakeMUDServer:
    """Minimal TCP server that mimics the Holodeck MUD protocol.

    Accepts one connection at a time, reads lines, and sends scripted
    responses.  Useful for integration tests without a real MUD.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self._port = port
        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._client_sock: Optional[socket.socket] = None
        self._sent_commands: list[str] = []
        # Responses: list of byte strings the server will send in order.
        self.responses: list[bytes] = []
        self._lock = threading.Lock()

    @property
    def port(self) -> int:
        if self._server_sock:
            return self._server_sock.getsockname()[1]
        return self._port

    def start(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self._port))
        self._server_sock.listen(5)
        self._server_sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        time.sleep(0.1)  # let it bind

    def stop(self) -> None:
        self._running = False
        if self._client_sock:
            try:
                self._client_sock.close()
            except OSError:
                pass
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_sent_commands(self) -> list[str]:
        with self._lock:
            return list(self._sent_commands)

    def _serve(self) -> None:
        while self._running:
            try:
                client, _ = self._server_sock.accept()  # type: ignore[union-attr]
                self._client_sock = client
                client.settimeout(1.0)
                self._handle_client(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_client(self, client: socket.socket) -> None:
        buf = b""
        resp_idx = 0
        # Send the first response immediately (welcome banner).
        if self.responses:
            client.sendall(self.responses[0])
            resp_idx = 1
        while self._running:
            try:
                data = client.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    cmd = line.decode("utf-8", errors="replace").strip()
                    with self._lock:
                        self._sent_commands.append(cmd)

                    # Send the next scripted response (if any).
                    if resp_idx < len(self.responses):
                        client.sendall(self.responses[resp_idx])
                        resp_idx += 1

            except socket.timeout:
                continue
            except OSError:
                break


# ===================================================================
# 1. Event System Tests
# ===================================================================

class TestMUDEvent(unittest.TestCase):
    """Test structured event creation, serialisation, and deserialisation."""

    def test_creation_defaults(self) -> None:
        event = MUDEvent(event_type=EventType.SYSTEM, raw="hello")
        self.assertEqual(event.event_type, EventType.SYSTEM)
        self.assertEqual(event.raw, "hello")
        self.assertTrue(len(event.event_id) == 12)
        self.assertIsInstance(event.timestamp, float)

    def test_to_dict_round_trip(self) -> None:
        event = MUDEvent(
            event_type=EventType.ROOM_ENTER,
            data={"room_name": "Docking Bay"},
            source="navigator",
            raw="--- Docking Bay ---",
        )
        d = event.to_dict()
        self.assertEqual(d["event_type"], "room_enter")
        self.assertEqual(d["data"]["room_name"], "Docking Bay")
        self.assertEqual(d["source"], "navigator")

        restored = MUDEvent.from_dict(d)
        self.assertEqual(restored.event_type, EventType.ROOM_ENTER)
        self.assertEqual(restored.event_id, event.event_id)
        self.assertEqual(restored.raw, event.raw)

    def test_str_representation(self) -> None:
        event = MUDEvent(event_type=EventType.ERROR, raw="something broke")
        self.assertIn("[error]", str(event))
        self.assertIn("something broke", str(event))


class TestEventQueue(unittest.TestCase):
    """Test the thread-safe event queue."""

    def test_push_and_drain(self) -> None:
        q = EventQueue()
        q.push(MUDEvent(event_type=EventType.SYSTEM, raw="a"))
        q.push(MUDEvent(event_type=EventType.MESSAGE, raw="b"))
        items = q.drain()
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].raw, "a")

        # Drain again → empty.
        self.assertEqual(q.drain(), [])

    def test_max_size(self) -> None:
        q = EventQueue(max_size=3)
        for i in range(5):
            q.push(MUDEvent(event_type=EventType.SYSTEM, raw=str(i)))
        self.assertEqual(q.size, 3)
        items = q.drain()
        # Oldest 2 should have been dropped.
        self.assertEqual(items[0].raw, "2")
        self.assertEqual(items[-1].raw, "4")

    def test_push_many(self) -> None:
        q = EventQueue()
        events = [MUDEvent(event_type=EventType.SYSTEM, raw=str(i)) for i in range(3)]
        q.push_many(events)
        self.assertEqual(q.size, 3)

    def test_clear(self) -> None:
        q = EventQueue()
        q.push(MUDEvent(event_type=EventType.SYSTEM, raw="x"))
        q.clear()
        self.assertEqual(q.size, 0)

    def test_peek_all_no_remove(self) -> None:
        q = EventQueue()
        q.push(MUDEvent(event_type=EventType.SYSTEM, raw="y"))
        snapshot = q.peek_all()
        self.assertEqual(len(snapshot), 1)
        # Should still be in the queue.
        self.assertEqual(q.size, 1)


class TestMUDParser(unittest.TestCase):
    """Test the raw-text → structured-event parser."""

    def setUp(self) -> None:
        self.parser = MUDParser()

    def test_whisper(self) -> None:
        events = self.parser.parse("[Alice] whispers: meet me at the bay")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.WHISPER)
        self.assertEqual(events[0].source, "Alice")
        self.assertEqual(events[0].data["text"], "meet me at the bay")

    def test_shout(self) -> None:
        events = self.parser.parse("Bob shouts: everyone to the bridge!")
        self.assertEqual(events[0].event_type, EventType.SHOUT)
        self.assertEqual(events[0].source, "Bob")

    def test_room_enter(self) -> None:
        events = self.parser.parse("--- Engineering Bay ---")
        self.assertEqual(events[0].event_type, EventType.ROOM_ENTER)
        self.assertEqual(events[0].data["room_name"], "Engineering Bay")

    def test_system_message(self) -> None:
        events = self.parser.parse("[System] Welcome to the Holodeck.")
        self.assertEqual(events[0].event_type, EventType.SYSTEM)
        self.assertEqual(events[0].data["text"], "Welcome to the Holodeck.")

    def test_combat_message(self) -> None:
        events = self.parser.parse("[Combat] You hit the drone for 12 damage.")
        self.assertEqual(events[0].event_type, EventType.COMBAT)

    def test_error_message(self) -> None:
        events = self.parser.parse("[Error] Command not recognized.")
        self.assertEqual(events[0].event_type, EventType.ERROR)

    def test_public_message(self) -> None:
        events = self.parser.parse("Charlie says: hello everyone")
        self.assertEqual(events[0].event_type, EventType.MESSAGE)
        self.assertEqual(events[0].source, "Charlie")

    def test_prompt(self) -> None:
        events = self.parser.parse("HP: 50/50 >")
        self.assertEqual(events[0].event_type, EventType.PROMPT)
        self.assertEqual(events[0].data["prompt"], "HP: 50/50 >")

    def test_unknown_line(self) -> None:
        events = self.parser.parse("some random text that doesn't match")
        self.assertEqual(events[0].event_type, EventType.UNKNOWN)

    def test_multiline_input(self) -> None:
        raw = "--- Bridge ---\n[Room] The main bridge.\nHP: 100/100 >"
        events = self.parser.parse(raw)
        types = [e.event_type for e in events]
        self.assertEqual(types, [EventType.ROOM_ENTER, EventType.ROOM_DESC, EventType.PROMPT])

    def test_blank_lines_ignored(self) -> None:
        events = self.parser.parse("\n\n[System] ok\n\n")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.SYSTEM)

    def test_convenience_factories(self) -> None:
        sys_ev = MUDParser.make_system("boot complete")
        self.assertEqual(sys_ev.event_type, EventType.SYSTEM)

        err_ev = MUDParser.make_error("oops")
        self.assertEqual(err_ev.event_type, EventType.ERROR)

        msg_ev = MUDParser.make_message("hi", source="bot")
        self.assertEqual(msg_ev.event_type, EventType.MESSAGE)
        self.assertEqual(msg_ev.source, "bot")


# ===================================================================
# 2. Session Manager Tests (mocked TCP)
# ===================================================================

class TestMUDSession(unittest.TestCase):
    """Test MUDSession with mocked TCP sockets."""

    def _make_mock_socket(self, recv_responses: list[bytes | BaseException]) -> MagicMock:
        """Create a mock socket.

        Args:
            recv_responses: Each item is either ``bytes`` (data returned by
                recv) or an exception instance (raised by recv).
                A trailing ``socket.timeout()`` is appended automatically
                so the reader loop can exit gracefully.
        """
        mock_sock = MagicMock()
        # Use a function that keeps returning socket.timeout after the
        # explicit responses are exhausted, avoiding StopIteration.
        _responses = list(recv_responses)
        def _recv_forever(*args: Any, **kwargs: Any) -> bytes:
            if _responses:
                val = _responses.pop(0)
                if isinstance(val, BaseException):
                    raise val
                return val
            raise socket.timeout()
        mock_sock.recv.side_effect = _recv_forever
        mock_sock.sendall = MagicMock()
        mock_sock.shutdown = MagicMock()
        mock_sock.close = MagicMock()
        mock_sock.settimeout = MagicMock()
        return mock_sock

    @patch("session.socket.socket")
    def test_connect_success(self, mock_socket_cls: MagicMock) -> None:
        """Session connects and reads welcome banner."""
        mock_socket_cls.return_value = self._make_mock_socket([
            b"[System] Welcome!\n",  # welcome banner (_read_raw call 1)
            b"--- Bridge ---\n",     # post-login (_read_raw call 2)
        ])
        session = MUDSession("sid1", "BotA", "engineer", host="127.0.0.1", port=7777)
        session.connect()

        self.assertTrue(session.connected)
        # Should have sent name and class.
        sent_data = [c.args[0] for c in mock_socket_cls.return_value.sendall.call_args_list]
        self.assertTrue(any(b"BotA" in d for d in sent_data))
        self.assertTrue(any(b"engineer" in d for d in sent_data))

        session.close()
        mock_socket_cls.return_value.shutdown.assert_called_once()

    @patch("session.socket.socket")
    def test_connect_failure(self, mock_socket_cls: MagicMock) -> None:
        """Session raises ConnectionError when MUD is unreachable."""
        mock_socket_cls.return_value.connect.side_effect = ConnectionRefusedError("refused")
        session = MUDSession("sid2", "BotB", "explorer")
        with self.assertRaises(ConnectionError):
            session.connect()
        self.assertFalse(session.connected)

    @patch("session.socket.socket")
    def test_send_command(self, mock_socket_cls: MagicMock) -> None:
        """Session.send writes to the socket."""
        mock_socket_cls.return_value = self._make_mock_socket([
            b"[System] ok\n",
            b"--- Room ---\n",
        ])
        session = MUDSession("sid3", "BotC", "engineer")
        session.connect()
        self.assertTrue(session.connected)
        session.send("look north")
        sent = mock_socket_cls.return_value.sendall.call_args_list[-1].args[0]
        self.assertIn(b"look north", sent)
        session.close()

    @patch("session.socket.socket")
    def test_send_when_disconnected_raises(self, mock_socket_cls: MagicMock) -> None:
        """Sending on a disconnected session raises RuntimeError."""
        session = MUDSession("sid4", "BotD", "explorer")
        with self.assertRaises(RuntimeError):
            session.send("look")
        # Also test after close.
        mock_socket_cls.return_value = self._make_mock_socket([
            b"[System] ok\n",
        ])
        session.connect()
        session.close()
        with self.assertRaises(RuntimeError):
            session.send("look")

    @patch("session.socket.socket")
    def test_recv_events_timeout(self, mock_socket_cls: MagicMock) -> None:
        """recv_events returns empty on timeout when no data available."""
        mock_sock = self._make_mock_socket([
            b"[System] ok\n",
        ])
        # Override: make recv always raise timeout after connect.
        mock_sock.recv.side_effect = socket.timeout()
        mock_socket_cls.return_value = mock_sock

        session = MUDSession("sid5", "BotE", "explorer")
        session.connect()
        # Use a short timeout for the test.
        events = session.event_queue.wait_for_events(timeout=0.1)
        self.assertEqual(len(events), 0)
        session.close()

    def test_is_idle(self) -> None:
        session = MUDSession("sid6", "BotF", "explorer")
        session.last_activity = time.time() - 200
        self.assertTrue(session.is_idle())
        session.last_activity = time.time()
        self.assertFalse(session.is_idle())

    def test_status_dict(self) -> None:
        session = MUDSession("sid7", "BotG", "scientist")
        d = session.status_dict()
        self.assertEqual(d["session_id"], "sid7")
        self.assertEqual(d["agent_name"], "BotG")
        self.assertEqual(d["agent_class"], "scientist")
        self.assertFalse(d["connected"])


# ===================================================================
# 3. Session Registry Tests
# ===================================================================

class TestSessionRegistry(unittest.TestCase):
    """Test the thread-safe session registry."""

    def setUp(self) -> None:
        self.registry = SessionRegistry()
        self.session = MUDSession("r1", "RegBot", "engineer")

    def test_add_and_get(self) -> None:
        self.registry.add(self.session)
        found = self.registry.get("r1")
        self.assertIsNotNone(found)
        self.assertEqual(found.agent_name, "RegBot")

    def test_get_missing(self) -> None:
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_remove(self) -> None:
        self.registry.add(self.session)
        removed = self.registry.remove("r1")
        self.assertEqual(removed.session_id, "r1")
        self.assertIsNone(self.registry.get("r1"))

    def test_find_by_name(self) -> None:
        self.registry.add(self.session)
        found = self.registry.find_by_name("regbot")  # case-insensitive
        self.assertEqual(found.session_id, "r1")
        self.assertIsNone(self.registry.find_by_name("nobody"))

    def test_all_sessions(self) -> None:
        self.registry.add(self.session)
        s2 = MUDSession("r2", "Bot2", "explorer")
        self.registry.add(s2)
        self.assertEqual(self.registry.count, 2)
        self.assertEqual(len(self.registry.all_sessions()), 2)


# ===================================================================
# 4. Bridge HTTP Endpoint Tests
# ===================================================================

class TestBridgeEndpointsNoMUD(unittest.TestCase):
    """Test bridge HTTP endpoints without a real MUD server.

    Tests that use only the bridge's own HTTP layer (no TCP to MUD).
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Start the bridge on a high port for testing."""
        cls.bridge_port = 19286
        cls.bridge = MudBridge(
            port=cls.bridge_port,
            mud_host="127.0.0.1",
            mud_port=7777,
        )
        cls.bridge.start(blocking=False)
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.bridge.stop()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.bridge_port}{path}"

    def _get(self, path: str) -> dict:
        req = Request(self._url(path))
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, data: dict) -> dict:
        payload = json.dumps(data).encode()
        req = Request(
            self._url(path),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())

    def _post_expect_error(self, path: str, data: dict, expected_status: int) -> dict:
        payload = json.dumps(data).encode()
        req = Request(
            self._url(path),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            self.assertEqual(exc.code, expected_status)
            return json.loads(exc.read().decode())

    def test_status_endpoint(self) -> None:
        result = self._get("/status")
        self.assertTrue(result["ok"])
        self.assertEqual(result["bridge"], "mud-http-bridge")
        self.assertIn("uptime_s", result)
        self.assertEqual(result["connected_agents"], 0)

    def test_connect_requires_agent_name(self) -> None:
        result = self._post_expect_error("/connect", {}, 400)
        self.assertFalse(result["ok"])
        self.assertIn("agent_name", result["error"])

    def test_connect_no_mud_server(self) -> None:
        """Connect should fail with 502 when no MUD is running."""
        result = self._post_expect_error(
            "/connect",
            {"agent_name": "TestBot", "agent_class": "engineer"},
            502,
        )
        self.assertFalse(result["ok"])

    def test_command_without_session(self) -> None:
        result = self._post_expect_error(
            "/command",
            {"session_id": "fake", "command": "look"},
            404,
        )
        self.assertFalse(result["ok"])

    def test_command_requires_fields(self) -> None:
        # Missing session_id.
        result = self._post_expect_error(
            "/command",
            {"command": "look"},
            400,
        )
        self.assertFalse(result["ok"])

        # Missing command.
        result = self._post_expect_error(
            "/command",
            {"session_id": "x"},
            400,
        )
        self.assertFalse(result["ok"])

    def test_output_without_session(self) -> None:
        """GET /output with a nonexistent session_id returns 404."""
        try:
            self._get("/output?session_id=nonexistent")
            self.fail("Expected HTTPError")
        except HTTPError as exc:
            self.assertEqual(exc.code, 404)
            body = json.loads(exc.read().decode())
            self.assertFalse(body["ok"])

    def test_disconnect_without_session(self) -> None:
        result = self._post_expect_error(
            "/disconnect",
            {"session_id": "nonexistent"},
            404,
        )
        self.assertFalse(result["ok"])

    def test_whisper_requires_fields(self) -> None:
        # Missing target.
        result = self._post_expect_error(
            "/whisper",
            {"session_id": "x", "message": "hi"},
            400,
        )
        self.assertFalse(result["ok"])

    def test_agents_empty(self) -> None:
        result = self._get("/agents")
        self.assertTrue(result["ok"])
        self.assertEqual(result["agents"], [])
        self.assertEqual(result["count"], 0)

    def test_rooms_empty(self) -> None:
        result = self._get("/rooms")
        self.assertTrue(result["ok"])
        self.assertEqual(result["rooms"], [])

    def test_unknown_route(self) -> None:
        req = Request(self._url("/nonexistent"))
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 404)

    def test_output_long_poll_disconnected_session(self) -> None:
        """GET /output with a non-connected session returns error event."""
        # Inject a dummy (not TCP-connected) session.
        session = MUDSession("poll-test", "PollBot", "engineer")
        self.bridge.registry.add(session)

        result = self._get("/output?session_id=poll-test&timeout=1")

        self.assertTrue(result["ok"])
        self.assertGreater(len(result["events"]), 0)
        self.assertEqual(result["events"][0]["event_type"], "error")
        self.assertFalse(result["connected"])

        self.bridge.registry.remove("poll-test")


# ===================================================================
# 5. Bridge Integration Tests (with fake MUD TCP server)
# ===================================================================

class TestBridgeWithFakeMUD(unittest.TestCase):
    """Integration tests: bridge ↔ fake MUD TCP server.

    Starts a lightweight TCP server that mimics the Holodeck MUD and
    verifies the full connect → command → disconnect cycle over real
    sockets (no mocks on the TCP layer).
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Start fake MUD.
        cls.fake_mud = FakeMUDServer()
        cls.fake_mud.responses = [
            b"[System] Welcome to the Holodeck MUD!\n",  # welcome banner
            b"--- Bridge ---\n[Room] The main bridge.\n",  # post-login
            b"[Look] You see a console.\n",  # response to 'look'
            b"[System] Whisper sent.\n",     # response to whisper
        ]
        cls.fake_mud.start()

        # Start bridge pointing at fake MUD.
        cls.bridge_port = 19287
        cls.bridge = MudBridge(
            port=cls.bridge_port,
            mud_host="127.0.0.1",
            mud_port=cls.fake_mud.port,
        )
        cls.bridge.start(blocking=False)
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.bridge.stop()
        cls.fake_mud.stop()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.bridge_port}{path}"

    def _get(self, path: str) -> dict:
        req = Request(self._url(path))
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, data: dict) -> dict:
        payload = json.dumps(data).encode()
        req = Request(
            self._url(path),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())

    def test_full_connect_command_disconnect_cycle(self) -> None:
        """End-to-end: connect, send command, whisper, disconnect."""
        # Connect.
        result = self._post("/connect", {
            "agent_name": "E2EBot",
            "agent_class": "engineer",
        })
        self.assertTrue(result["ok"])
        session_id = result["session_id"]
        self.assertTrue(result["connected"])
        self.assertIn("initial_events", result)

        # Check agents list.
        agents = self._get("/agents")
        self.assertEqual(agents["count"], 1)
        self.assertEqual(agents["agents"][0]["agent_name"], "E2EBot")

        # Send a command.
        cmd_result = self._post("/command", {
            "session_id": session_id,
            "command": "look",
        })
        self.assertTrue(cmd_result["ok"])
        self.assertEqual(cmd_result["command"], "look")

        # Whisper.
        w = self._post("/whisper", {
            "session_id": session_id,
            "target": "OtherBot",
            "message": "secret message",
        })
        self.assertTrue(w["ok"])
        self.assertEqual(w["target"], "OtherBot")
        self.assertEqual(w["whisper"], "secret message")

        # Give a moment for commands to flush over TCP.
        time.sleep(0.3)

        # Verify the fake MUD received the expected commands.
        sent = self.fake_mud.get_sent_commands()
        self.assertTrue(any("E2EBot" in c for c in sent))  # name
        self.assertTrue(any("engineer" in c for c in sent))  # class
        self.assertTrue(any("look" == c.strip() for c in sent))  # command
        self.assertTrue(any("whisper" in c for c in sent))  # whisper

        # Disconnect.
        disc_result = self._post("/disconnect", {"session_id": session_id})
        self.assertTrue(disc_result["ok"])
        self.assertTrue(disc_result["disconnected"])

        # Verify session removed.
        agents_after = self._get("/agents")
        self.assertEqual(agents_after["count"], 0)

    def test_connect_duplicate_name(self) -> None:
        """Connecting the same name twice should fail with 409."""
        r1 = self._post("/connect", {
            "agent_name": "DupBot",
            "agent_class": "explorer",
        })
        self.assertTrue(r1["ok"])
        sid1 = r1["session_id"]

        try:
            payload = json.dumps({
                "agent_name": "DupBot",
                "agent_class": "explorer",
            }).encode()
            req = Request(
                self._url("/connect"),
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as ctx:
                urlopen(req, timeout=5)
            self.assertEqual(ctx.exception.code, 409)
        finally:
            # Cleanup.
            self._post("/disconnect", {"session_id": sid1})


# ===================================================================
# 6. Graceful Shutdown Test
# ===================================================================

class TestGracefulShutdown(unittest.TestCase):
    """Test that bridge stops cleanly and sessions are closed."""

    def test_stop_closes_all_sessions(self) -> None:
        bridge = MudBridge(port=19288)
        # Inject mock sessions.
        s1 = MUDSession("gs1", "ShutdownBot1", "engineer")
        s2 = MUDSession("gs2", "ShutdownBot2", "explorer")
        bridge.registry.add(s1)
        bridge.registry.add(s2)

        bridge.start(blocking=False)
        time.sleep(0.2)
        bridge.stop()

        # Sessions should have been closed (their connected flag set to False).
        self.assertFalse(s1.connected)
        self.assertFalse(s2.connected)


if __name__ == "__main__":
    unittest.main()
