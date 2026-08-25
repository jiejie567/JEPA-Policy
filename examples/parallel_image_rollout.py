"""Experimental process-parallel image rollout smoke test.

This entry point is intentionally separate from train_robomimic.py. The default
serial training evaluation path is unchanged.
"""

import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import hydra
import numpy as np
from omegaconf import OmegaConf


def _rollout_worker(config_dict, worker_id, episode_ids, result_queue):
    worker_started = time.perf_counter()
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        # Keep MUJOCO_EGL_DEVICE_ID unset. Software Mesa can instead be pinned
        # with JEPA_POLICY_EGL_DEVICE_ID without changing CUDA card selection.

        import torch

        from jepa_policy.agent import TrainingAgent
        from jepa_policy.datasets.robot_dataset import make_dataset
        from jepa_policy.envs.robot_env import make_vec_env
        from jepa_policy.eval_rng import get_rollout_seed
        from jepa_policy.torch_utils import set_seed

        config = OmegaConf.create(config_dict)
        config.task.num_envs = 1
        config.task.save_video = False
        config.log.save_video = False
        config.log.wandb_mode = "disabled"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside rollout worker")
        torch.cuda.set_device(0)

        rollout_seed = get_rollout_seed(config)
        envs = make_vec_env(
            config.task,
            seed=rollout_seed + int(episode_ids[0]),
        )
        dataset = make_dataset(config.task, mode="train")
        agent = TrainingAgent(config)
        agent.load(config.optimization.model_path, load_optimizer=False)
        agent.eval()

        startup_seconds = time.perf_counter() - worker_started
        rollout_started = time.perf_counter()
        records = []

        for episode_id in episode_ids:
            episode_seed = rollout_seed + int(episode_id)
            set_seed(episode_seed)
            try:
                obs, _ = envs.reset(seed=episode_seed)
            except TypeError:
                obs, _ = envs.reset()

            episode_return = np.zeros(1, dtype=np.float32)
            length = 0
            episode_started = time.perf_counter()

            while length < int(config.task.max_episode_steps):
                obs_raw = obs
                obs = {}
                for key, value in obs_raw.items():
                    value = value.astype(np.float32)
                    value = dataset.normalizer["obs"][key].normalize(value)
                    obs[key] = torch.tensor(
                        value,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    )

                act_0 = torch.zeros(
                    (
                        1,
                        int(config.task.horizon),
                        int(config.task.act_dim),
                    ),
                    device=config.optimization.device,
                )
                joint_rollout = bool(
                    getattr(config.optimization, "future_joint_mode", False)
                )
                sample_output = agent.sample(
                    act_0=act_0,
                    obs=obs,
                    num_steps=int(config.eval.num_steps),
                    use_ema=True,
                    return_future=joint_rollout,
                )
                if joint_rollout:
                    act_normed, future_pred_1 = sample_output
                else:
                    act_normed = sample_output
                act = dataset.normalizer["action"].unnormalize(
                    act_normed.detach().cpu().numpy()
                )
                start = int(config.task.obs_steps) - 1
                end = start + int(config.task.act_steps)
                act = act[:, start:end, :]

                if bool(config.task.abs_action):
                    act = dataset.undo_transform_action(act)

                obs, reward, terminated, truncated, _ = envs.step(act)
                episode_return += reward
                length += int(config.task.act_steps)

            records.append(
                {
                    "episode_id": int(episode_id),
                    "seed": episode_seed,
                    "worker_id": int(worker_id),
                    "success": float(episode_return[0] > 0),
                    "return": float(episode_return[0]),
                    "length": int(length),
                    "seconds": time.perf_counter() - episode_started,
                }
            )

        rollout_seconds = time.perf_counter() - rollout_started
        max_cuda_memory = int(torch.cuda.max_memory_allocated())
        envs.close()
        result_queue.put(
            {
                "worker_id": int(worker_id),
                "status": "ok",
                "startup_seconds": startup_seconds,
                "rollout_seconds": rollout_seconds,
                "max_cuda_memory_bytes": max_cuda_memory,
                "records": records,
            }
        )
    except Exception:
        result_queue.put(
            {
                "worker_id": int(worker_id),
                "status": "error",
                "traceback": traceback.format_exc(),
            }
        )


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    if not bool(config.eval.parallel_rollout):
        raise RuntimeError(
            "Experimental rollout is disabled. Set eval.parallel_rollout=true."
        )
    if config.task.obs_type != "image":
        raise ValueError("parallel_image_rollout.py only supports image tasks")
    if not config.optimization.model_path:
        raise ValueError("optimization.model_path must point to a trained checkpoint")
    if not Path(config.optimization.model_path).is_file():
        raise FileNotFoundError(config.optimization.model_path)

    workers = int(config.eval.parallel_rollout_workers)
    episodes_per_worker = int(config.eval.episodes_per_worker)
    if workers < 1 or episodes_per_worker < 1:
        raise ValueError("workers and episodes_per_worker must both be positive")

    config.task.save_video = False
    config.log.save_video = False
    config.log.wandb_mode = "disabled"

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    started = time.perf_counter()

    for worker_id in range(workers):
        first_episode = worker_id * episodes_per_worker
        episode_ids = list(
            range(first_episode, first_episode + episodes_per_worker)
        )
        process = context.Process(
            target=_rollout_worker,
            args=(
                OmegaConf.to_container(config, resolve=True),
                worker_id,
                episode_ids,
                result_queue,
            ),
            name=f"image-rollout-{worker_id}",
        )
        process.start()
        processes.append(process)

    timeout_seconds = float(config.eval.worker_timeout_seconds)
    deadline = time.perf_counter() + timeout_seconds
    for process in processes:
        process.join(max(0.0, deadline - time.perf_counter()))

    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
        process.join(10)

    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty:
            break

    total_seconds = time.perf_counter() - started
    errors = [result for result in results if result["status"] != "ok"]
    missing_workers = sorted(
        set(range(workers)) - {result["worker_id"] for result in results}
    )
    if timed_out or errors or missing_workers:
        details = {
            "timed_out_workers": [p.name for p in timed_out],
            "missing_workers": missing_workers,
            "errors": errors,
            "exit_codes": {p.name: p.exitcode for p in processes},
        }
        raise RuntimeError(json.dumps(details, indent=2))

    records = sorted(
        [
            record
            for result in results
            for record in result["records"]
        ],
        key=lambda record: record["episode_id"],
    )
    summary = {
        "workers": workers,
        "episodes_per_worker": episodes_per_worker,
        "eval_episodes": len(records),
        "base_seed": int(config.eval.rollout_seed),
        "total_eval_seconds": total_seconds,
        "mean_success": float(np.mean([r["success"] for r in records])),
        "mean_return": float(np.mean([r["return"] for r in records])),
        "mean_length": float(np.mean([r["length"] for r in records])),
        "workers_detail": results,
        "episodes": records,
    }

    output_path = Path(config.eval.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
