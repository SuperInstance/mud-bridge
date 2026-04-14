"""
cli.py — MUD Bridge CLI

Command-line interface for the MUD HTTP Bridge.  Provides subcommands for
starting the server, checking status, listing agents, testing MUD connectivity,
and performing initial onboarding.

Usage::

    python cli.py serve              # Start bridge on :8877
    python cli.py serve --port 9000  # Custom port
    python cli.py status             # Show bridge status
    python cli.py list-agents        # List connected agents
    python cli.py test               # Test MUD server connectivity
    python cli.py onboard            # First-time setup
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

DEFAULT_BRIDGE_URL = "http://127.0.0.1:8877"


def _http_get(path: str, bridge_url: str = DEFAULT_BRIDGE_URL) -> dict:
    """Perform a GET request against the bridge."""
    url = f"{bridge_url}{path}"
    try:
        req = Request(url)
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "body": body}
    except (URLError, OSError) as exc:
        return {"ok": False, "error": f"Cannot reach bridge at {bridge_url}: {exc}"}


def _http_post(
    path: str,
    data: dict,
    bridge_url: str = DEFAULT_BRIDGE_URL,
) -> dict:
    """Perform a POST request against the bridge."""
    url = f"{bridge_url}{path}"
    try:
        payload = json.dumps(data).encode("utf-8")
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "body": body}
    except (URLError, OSError) as exc:
        return {"ok": False, "error": f"Cannot reach bridge at {bridge_url}: {exc}"}


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    """Start the MUD bridge server."""
    _setup_logging(verbose=args.verbose)

    from bridge import MudBridge

    bridge = MudBridge(
        port=args.port,
        mud_host=args.mud_host,
        mud_port=args.mud_port,
    )
    bridge.start(blocking=True)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Query the running bridge for its status."""
    _setup_logging()
    result = _http_get("/status", bridge_url=args.url)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def cmd_list_agents(args: argparse.Namespace) -> int:
    """List all connected agents."""
    _setup_logging()
    result = _http_get("/agents", bridge_url=args.url)
    if not result.get("ok"):
        print(json.dumps(result, indent=2))
        return 1

    agents = result.get("agents", [])
    print(f"Connected agents: {result.get('count', 0)}\n")
    for agent in agents:
        status = "●" if agent.get("connected") else "○"
        print(
            f"  {status} {agent.get('agent_name', '?')} "
            f"({agent.get('agent_class', '?')})  "
            f"session={agent.get('session_id', '?')}  "
            f"events={agent.get('pending_events', 0)}"
        )
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Test TCP connectivity to the MUD server."""
    _setup_logging()
    host = args.mud_host
    port = args.mud_port

    print(f"Testing TCP connection to {host}:{port} …")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        start = time.time()
        sock.connect((host, port))
        elapsed = (time.time() - start) * 1000
        sock.close()
        print(f"  ✓ Connected in {elapsed:.1f} ms")
        return 0
    except ConnectionRefusedError:
        print(f"  ✗ Connection refused — is the MUD server running on {host}:{port}?")
        return 1
    except socket.timeout:
        print(f"  ✗ Connection timed out after 5 s")
        return 1
    except OSError as exc:
        print(f"  ✗ Error: {exc}")
        return 1


def cmd_onboard(args: argparse.Namespace) -> int:
    """Perform first-time onboarding checks."""
    _setup_logging(verbose=args.verbose)
    print("=== MUD Bridge Onboarding ===\n")

    # 1. Test MUD connectivity.
    print("[1/3] MUD server connectivity …")
    host = args.mud_host
    port = args.mud_port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        sock.close()
        print(f"  ✓ MUD server reachable at {host}:{port}\n")
    except OSError as exc:
        print(f"  ✗ Cannot reach MUD at {host}:{port}: {exc}")
        print("  → Start the MUD server first, then re-run onboard.\n")

    # 2. Check if bridge port is available.
    print("[2/3] Bridge port availability …")
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_sock.bind(("0.0.0.0", args.port))
        test_sock.close()
        print(f"  ✓ Port {args.port} is available\n")
    except OSError as exc:
        print(f"  ✗ Port {args.port} is in use: {exc}\n")

    # 3. Module import check.
    print("[3/3] Module imports …")
    try:
        from events import MUDEvent, MUDParser, EventQueue  # noqa: F401
        print("  ✓ events module OK")
    except ImportError as exc:
        print(f"  ✗ events module failed: {exc}")
    try:
        from session import MUDSession  # noqa: F401
        print("  ✓ session module OK")
    except ImportError as exc:
        print(f"  ✗ session module failed: {exc}")
    try:
        from bridge import MudBridge  # noqa: F401
        print("  ✓ bridge module OK")
    except ImportError as exc:
        print(f"  ✗ bridge module failed: {exc}")

    print("\nOnboarding complete.  Run `python cli.py serve` to start the bridge.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mud-bridge",
        description="MUD HTTP Bridge — programmatic agent↔MUD connection via HTTP",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--mud-host",
        default="127.0.0.1",
        help="MUD server hostname (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--mud-port",
        type=int,
        default=7777,
        help="MUD server port (default: 7777)",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    p_serve = sub.add_parser("serve", help="Start the bridge server")
    p_serve.add_argument("--port", type=int, default=8877, help="Bridge port")
    p_serve.set_defaults(func=cmd_serve)

    # status
    p_status = sub.add_parser("status", help="Show bridge status")
    p_status.add_argument("--url", default=DEFAULT_BRIDGE_URL, help="Bridge URL")
    p_status.set_defaults(func=cmd_status)

    # list-agents
    p_agents = sub.add_parser("list-agents", help="List connected agents")
    p_agents.add_argument("--url", default=DEFAULT_BRIDGE_URL, help="Bridge URL")
    p_agents.set_defaults(func=cmd_list_agents)

    # test
    p_test = sub.add_parser("test", help="Test MUD server connectivity")
    p_test.set_defaults(func=cmd_test)

    # onboard
    p_onboard = sub.add_parser("onboard", help="First-time setup checks")
    p_onboard.add_argument("--port", type=int, default=8877, help="Bridge port")
    p_onboard.set_defaults(func=cmd_onboard)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
