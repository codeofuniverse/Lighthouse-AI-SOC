"""WebSocket connection manager — broadcasts alerts to all connected clients."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.debug("WS connected, total=%d", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        logger.debug("WS disconnected, total=%d", len(self._connections))

    async def broadcast(self, data: dict[str, Any]) -> None:
        dead: set[WebSocket] = set()
        for ws in list(self._connections):
            try:
                await ws.send_json(data, mode="text")
            except Exception:
                dead.add(ws)
        self._connections -= dead


ws_manager = ConnectionManager()
