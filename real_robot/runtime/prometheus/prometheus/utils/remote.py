from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from typing import Any

import numpy as np

DEFAULT_MAX_SIZE = 128 * 1024 * 1024
TUNNEL_TIMEOUT_S = 15.0


class RemoteInferenceClient:
    """One-request-at-a-time websocket client for remote policy servers."""

    def __init__(
        self,
        server: str,
        *,
        max_size: int = DEFAULT_MAX_SIZE,
        ssh_tunnel: dict[str, Any] | None = None,
    ):
        server = str(server)
        self.server = server if server.startswith("ws") else f"ws://{server}"
        self.max_size = int(max_size)
        self.ssh_tunnel = dict(ssh_tunnel or {})
        self._tunnel_proc: subprocess.Popen[Any] | None = None
        self._ws: Any | None = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            self._connect()
        self._ws.send(pack_msg(payload))
        raw = self._ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(raw)
        response = unpack_msg(raw)
        if not isinstance(response, dict):
            raise ValueError(f"remote response must be a dict, got {type(response).__name__}")
        return response

    def close(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        finally:
            self._ws = None
            self._close_tunnel()

    def _connect(self) -> None:
        from websockets.sync.client import connect

        self._ensure_tunnel()
        self._ws = connect(self.server, compression=None, max_size=self.max_size)

    def _ensure_tunnel(self) -> None:
        if not self.ssh_tunnel.get("enabled", False) or self._tunnel_proc is not None:
            return
        local_port = int(self.ssh_tunnel["local_port"])
        if bool(self.ssh_tunnel.get("replace_existing", True)):
            _close_local_port_listener(local_port)
        proc = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-L",
                f"{local_port}:{self.ssh_tunnel.get('remote_host', '127.0.0.1')}:{int(self.ssh_tunnel['remote_port'])}",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "ExitOnForwardFailure=yes",
                "-p",
                str(int(self.ssh_tunnel.get("ssh_port", 22))),
                str(self.ssh_tunnel["ssh_host"]),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", "")},
        )
        _wait_for_local_port(local_port, proc)
        self._tunnel_proc = proc

    def _close_tunnel(self) -> None:
        proc, self._tunnel_proc = self._tunnel_proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


def pack_msg(obj: Any) -> bytes:
    import msgpack

    return msgpack.packb(_encode(obj), use_bin_type=True)


def unpack_msg(data: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(data, raw=False, object_hook=_decode)


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__ndarray__": True, "dtype": str(arr.dtype), "shape": arr.shape, "data": arr.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(key): _encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(value) for value in obj]
    return obj


def _decode(obj: Any) -> Any:
    if not isinstance(obj, dict) or not obj.get("__ndarray__"):
        return obj
    return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(tuple(obj["shape"])).copy()


def _wait_for_local_port(port: int, proc: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + TUNNEL_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(f"ssh tunnel exited with code {proc.returncode}: {stderr.strip()}")
        if _local_port_open(port):
            return
        time.sleep(0.05)
    proc.terminate()
    raise TimeoutError(f"timed out waiting for ssh tunnel on localhost:{port}")


def _local_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.1):
            return True
    except OSError:
        return False


def _close_local_port_listener(port: int) -> None:
    output = _run(["lsof", f"-tiTCP:{int(port)}", "-sTCP:LISTEN"])
    for item in [] if output is None else output.split():
        if item.isdigit():
            os.kill(int(item), signal.SIGTERM)


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
