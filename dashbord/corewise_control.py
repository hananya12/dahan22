"""
corewise_control.py
--------------------
Shared real-time communication layer between the Corewise AI engine
(main.py) and the Corewise Dashboard (app.py).

Both sides talk to `corewise_server.py` over a persistent WebSocket
connection. No JSON / TXT / CSV file is used for synchronization - all
state changes are pushed the instant they happen.

`CorewiseClient` runs its own asyncio event loop on a background daemon
thread, so it can be dropped into either:
  * main.py     -> a synchronous OpenCV frame loop, or
  * app.py      -> a synchronous Streamlit script

...without either of them needing to become "async" themselves. Callers
interact with it through simple, thread-safe methods:

    client = CorewiseClient(role="engine")
    client.start()

    client.send("telemetry", {"tam": 12, "sam": 4})
    latest_control = client.get_state()          # dict, always current
    connected = client.is_connected()

Automatic reconnection is built in, so a server restart never crashes
the engine or the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

import websockets

DEFAULT_SERVER_URL = "ws://localhost:8765"
RECONNECT_DELAY_SECONDS = 1.5


class CorewiseClient:
    """Background-thread WebSocket client for engine or dashboard roles."""

    def __init__(
        self,
        role: str,
        server_url: str = DEFAULT_SERVER_URL,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Args:
            role: "engine" or "dashboard".
            server_url: WebSocket URL of corewise_server.py.
            on_message: optional callback invoked with every full
                {"type": ..., "payload": ...} message received, in
                addition to it being merged into the internal state.
        """
        if role not in ("engine", "dashboard"):
            raise ValueError("role must be 'engine' or 'dashboard'")

        self.role = role
        self.server_url = server_url
        self.on_message = on_message

        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}
        self._events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._send_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._started = False

    # ------------------------------------------------------------------
    # Public, thread-safe API
    # ------------------------------------------------------------------

    def start(self) -> "CorewiseClient":
        """Start the background connection thread (idempotent)."""
        if not self._started:
            self._started = True
            self._thread.start()
        return self

    def send(self, msg_type: str, payload: Dict[str, Any]) -> None:
        """Queue a message to be sent to the server as soon as possible."""
        self._send_queue.put({"type": msg_type, "payload": payload})

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the latest state received from the server."""
        with self._lock:
            return dict(self._state)

    def pop_events(self) -> list[Dict[str, Any]]:
        """Drain and return any live events (e.g. person entered/exited)."""
        drained = []
        while True:
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                break
        return drained

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Internal asyncio machinery (runs on the background thread)
    # ------------------------------------------------------------------

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connection_loop())
        except Exception:
            pass

    async def _connection_loop(self) -> None:
        register_msg = json.dumps({"type": "register", "role": self.role})

        while True:
            try:
                async with websockets.connect(self.server_url) as ws:
                    await ws.send(register_msg)
                    self._connected = True

                    sender = asyncio.create_task(self._sender_task(ws))
                    receiver = asyncio.create_task(self._receiver_task(ws))

                    done, pending = await asyncio.wait(
                        [sender, receiver], return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()

            except Exception:
                pass

            self._connected = False
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)

    async def _sender_task(self, ws: "websockets.WebSocketClientProtocol") -> None:
        loop = asyncio.get_event_loop()
        while True:
            message = await loop.run_in_executor(None, self._send_queue.get)
            await ws.send(json.dumps(message))

    async def _receiver_task(self, ws: "websockets.WebSocketClientProtocol") -> None:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            payload = data.get("payload", {})
            msg_type = data.get("type", "")

            with self._lock:
                self._state.update(payload)
                self._state["_connected"] = True
                self._state["_last_message_at"] = time.time()

            if msg_type == "event":
                self._events.put(payload)

            if self.on_message:
                self.on_message(data)
