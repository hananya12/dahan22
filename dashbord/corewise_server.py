"""
corewise_server.py
-------------------
Central real-time hub for Corewise.

Architecture:

    Dashboard (Streamlit, N instances)
        ⇅ WebSocket
    Corewise Server   (this file)
        ⇅ WebSocket
    AI Engine (main.py, 1..N cameras in the future)

The server keeps the current system state ENTIRELY IN MEMORY (no JSON /
TXT / CSV files are used for synchronization, per architecture spec) and
relays messages between the two sides the moment they arrive:

  * When the AI engine sends a "telemetry" or "event" message, it is
    broadcast immediately to every connected dashboard.
  * When a dashboard sends a "command" message, it is broadcast
    immediately to every connected AI engine.

Any client that connects late is caught up instantly: engines receive
the last known control state, dashboards receive the last known
telemetry state.

This design already supports the future requirements without an
architecture change:
  * Multiple engines (multiple cameras / stores) -> just tag each
    connection with a `client_id` and route/aggregate by that id.
  * Multiple dashboards / multiple concurrent users -> the server
    already fans out to a set of dashboard connections.
  * REST API / Mobile app -> can be added as another consumer of the
    same in-memory `SystemState`, no change needed to the engine or
    the dashboard.

Run with:
    python corewise_server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [SERVER] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("corewise_server")

HOST = "localhost"
PORT = 8765


@dataclass
class SystemState:
    """In-memory, thread-safe-by-design (single asyncio loop) system state."""

    # Latest telemetry pushed by the AI engine.
    telemetry: Dict[str, Any] = field(
        default_factory=lambda: {
            "tam": 0,
            "sam": 0,
            "som": 0,
            "inside": 0,
            "average_stay_time": 0.0,
            "conversion_rate": 0.0,
            "people_today": 0,
            "fps": 0.0,
            "camera_status": "offline",
            "current_model": "yolov8n.pt",
            "current_confidence": 0.5,
            "last_update": 0.0,
        }
    )

    # Latest control state requested by the dashboard.
    control: Dict[str, Any] = field(
        default_factory=lambda: {
            "blur_faces": False,
            "night_vision": False,
            "camera_running": True,
            "detection_paused": False,
            "confidence": 0.5,
            "model": "yolov8n.pt",
        }
    )

    def update_telemetry(self, payload: Dict[str, Any]) -> None:
        self.telemetry.update(payload)
        self.telemetry["last_update"] = time.time()

    def update_control(self, payload: Dict[str, Any]) -> None:
        self.control.update(payload)


class ConnectionHub:
    """Tracks connected engines / dashboards and fans out messages."""

    def __init__(self) -> None:
        self.engines: Set[WebSocketServerProtocol] = set()
        self.dashboards: Set[WebSocketServerProtocol] = set()
        self.state = SystemState()

    async def register(self, ws: WebSocketServerProtocol, role: str) -> None:
        if role == "engine":
            self.engines.add(ws)
            log.info("Engine connected (%d total).", len(self.engines))
            await self._send(ws, "control_sync", self.state.control)
        else:
            self.dashboards.add(ws)
            log.info("Dashboard connected (%d total).", len(self.dashboards))
            await self._send(ws, "telemetry_sync", self.state.telemetry)

    async def unregister(self, ws: WebSocketServerProtocol) -> None:
        self.engines.discard(ws)
        self.dashboards.discard(ws)
        log.info(
            "Client disconnected (engines=%d, dashboards=%d).",
            len(self.engines),
            len(self.dashboards),
        )
        if not self.engines:
            self.state.update_telemetry({"camera_status": "offline"})
            await self.broadcast_to_dashboards("telemetry_sync", self.state.telemetry)

    @staticmethod
    async def _send(ws: WebSocketServerProtocol, msg_type: str, payload: Dict[str, Any]) -> None:
        try:
            await ws.send(json.dumps({"type": msg_type, "payload": payload}))
        except websockets.ConnectionClosed:
            pass

    async def broadcast_to_dashboards(self, msg_type: str, payload: Dict[str, Any]) -> None:
        if not self.dashboards:
            return
        dead = []
        for ws in self.dashboards:
            try:
                await self._send(ws, msg_type, payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.dashboards.discard(ws)

    async def broadcast_to_engines(self, msg_type: str, payload: Dict[str, Any]) -> None:
        if not self.engines:
            return
        dead = []
        for ws in self.engines:
            try:
                await self._send(ws, msg_type, payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.engines.discard(ws)

    async def handle_message(self, ws: WebSocketServerProtocol, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Dropped malformed message.")
            return

        msg_type = data.get("type")
        payload = data.get("payload", {})

        if msg_type in ("telemetry", "event"):
            # Coming from the AI engine -> update state, fan out to dashboards.
            self.state.update_telemetry(payload)
            await self.broadcast_to_dashboards(
                "telemetry" if msg_type == "telemetry" else "event", payload
            )

        elif msg_type == "command":
            # Coming from a dashboard -> update state, fan out to engines.
            self.state.update_control(payload)
            await self.broadcast_to_engines("command", payload)


HUB = ConnectionHub()


async def handler(ws: WebSocketServerProtocol) -> None:
    """Per-connection entry point. First message must be a `register`."""
    role: Optional[str] = None
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        data = json.loads(raw)
        if data.get("type") != "register":
            await ws.close(reason="Expected register message first.")
            return
        role = data.get("role", "dashboard")
        await HUB.register(ws, role)

        async for raw in ws:
            await HUB.handle_message(ws, raw)

    except (websockets.ConnectionClosed, asyncio.TimeoutError):
        pass
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Handler error: %s", exc)
    finally:
        await HUB.unregister(ws)


async def main() -> None:
    log.info("Corewise Server starting on ws://%s:%d", HOST, PORT)
    async with websockets.serve(handler, HOST, PORT, ping_interval=20, ping_timeout=20):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server stopped.")
