"""Persistent env-only process pool for experimental image rollout evaluation."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from copy import deepcopy

from omegaconf import OmegaConf


def _env_worker(task_config_dict, worker_id, base_seed, connection):
    """Own exactly one MuJoCo/EGL env; never create a model or CUDA tensor."""
    env = None
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        # Keep MUJOCO_EGL_DEVICE_ID unset. The launcher uses the separate
        # JEPA_POLICY_EGL_DEVICE_ID setting because Mesa's EGL index is not the
        # physical CUDA_VISIBLE_DEVICES id.

        from mip.envs.robot_env import make_vec_env

        task_config = OmegaConf.create(task_config_dict)
        task_config.num_envs = 1
        task_config.save_video = False
        env = make_vec_env(task_config, seed=int(base_seed) + int(worker_id))
        connection.send(("ready", {"worker_id": int(worker_id)}))

        while True:
            command, payload = connection.recv()
            if command == "reset":
                seed = int(payload["seed"])
                try:
                    result = env.reset(seed=seed)
                except TypeError:
                    result = env.reset()
                connection.send(("ok", result))
            elif command == "step":
                connection.send(("ok", env.step(payload["action"])))
            elif command == "close":
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"Unknown rollout worker command: {command}")
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        connection.close()


class PersistentImageRolloutPool:
    """Persistent spawn workers containing only image environments."""

    def __init__(self, config, lazy=False):
        self.num_workers = int(config.eval.parallel_rollout_workers)
        self.timeout_seconds = float(config.eval.worker_timeout_seconds)
        if self.num_workers < 1:
            raise ValueError("eval.parallel_rollout_workers must be positive")

        task_config = deepcopy(config.task)
        task_config.num_envs = 1
        task_config.save_video = False
        self.task_dict = OmegaConf.to_container(task_config, resolve=True)
        self.base_seed = int(getattr(config.eval, "rollout_seed", 12345))
        self.connections = []
        self.processes = []
        self.startup_seconds = 0.0
        self.started = False
        if not lazy:
            self.start()

    def start(self):
        """Start rollout workers once, immediately before their first use."""
        if self.started:
            return
        context = mp.get_context("spawn")
        started = time.perf_counter()

        for worker_id in range(self.num_workers):
            parent, child = context.Pipe()
            process = context.Process(
                target=_env_worker,
                args=(
                    self.task_dict,
                    worker_id,
                    self.base_seed,
                    child,
                ),
                name=f"persistent-image-env-{worker_id}",
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

        for worker_id, connection in enumerate(self.connections):
            status, payload = self._recv(connection, worker_id)
            if status != "ready":
                self.close()
                raise RuntimeError(
                    f"Image rollout worker {worker_id} failed to start: {payload}"
                )
        self.startup_seconds = time.perf_counter() - started
        self.started = True

    def _recv(self, connection, worker_id):
        if not connection.poll(self.timeout_seconds):
            raise TimeoutError(f"Image rollout worker {worker_id} timed out")
        status, payload = connection.recv()
        if status == "error":
            raise RuntimeError(f"Image rollout worker {worker_id} failed:\n{payload}")
        return status, payload

    def reset(self, seeds):
        self.start()
        if len(seeds) != self.num_workers:
            raise ValueError("reset seeds must match parallel worker count")
        for connection, seed in zip(self.connections, seeds, strict=True):
            connection.send(("reset", {"seed": int(seed)}))
        return [
            self._recv(connection, worker_id)[1]
            for worker_id, connection in enumerate(self.connections)
        ]

    def step(self, actions, worker_indices=None):
        self.start()
        if worker_indices is None:
            worker_indices = list(range(self.num_workers))
        else:
            worker_indices = [int(index) for index in worker_indices]
        if len(actions) != len(worker_indices):
            raise ValueError("actions must match selected rollout workers")
        if len(set(worker_indices)) != len(worker_indices) or any(
            index < 0 or index >= self.num_workers
            for index in worker_indices
        ):
            raise ValueError("worker_indices must be unique valid workers")
        for worker_index, action in zip(
            worker_indices, actions, strict=True
        ):
            connection = self.connections[worker_index]
            connection.send(("step", {"action": action}))
        return [
            self._recv(self.connections[worker_index], worker_index)[1]
            for worker_index in worker_indices
        ]

    def close(self):
        for connection in getattr(self, "connections", []):
            try:
                connection.send(("close", None))
            except Exception:
                pass
        for worker_id, connection in enumerate(getattr(self, "connections", [])):
            try:
                if connection.poll(2):
                    connection.recv()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
        for process in getattr(self, "processes", []):
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
        self.connections = []
        self.processes = []
        self.started = False
