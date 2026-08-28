from __future__ import annotations

from typing import Any


class ZmqEnvServer:
    """ZMQ dict request/reply transport for external policy clients."""

    def __init__(self, *, addr: str):
        self.addr = str(addr)
        self._socket: Any | None = None

    def __enter__(self) -> "ZmqEnvServer":
        import zmq

        self._socket = zmq.Context.instance().socket(zmq.REP)
        self._socket.bind(self.addr)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def recv(self) -> dict[str, Any]:
        assert self._socket is not None
        request = self._socket.recv_pyobj()
        if not isinstance(request, dict):
            raise TypeError(f"ZMQ env request must be dict, got {type(request).__name__}")
        return request

    def reply(self, request_id: str, **payload: Any) -> None:
        self._send({"ok": True, "request_id": str(request_id), **payload})

    def reject(self, request_id: str, message: str) -> None:
        self._send({"ok": False, "request_id": str(request_id), "message": str(message)})

    def _send(self, payload: dict[str, Any]) -> None:
        assert self._socket is not None
        self._socket.send_pyobj(payload)
