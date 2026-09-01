import collections
import math
import pathlib

import dill
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.libero.libero_image_wrapper import LiberoImageWrapper
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class LiberoImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        libero_root,
        bddl_file,
        init_states_file,
        shape_meta,
        n_test=10,
        n_test_vis=2,
        test_start_seed=0,
        max_steps=300,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_rgb",
        camera_size=128,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        n_envs=2,
    ):
        super().__init__(output_dir)
        n_envs = min(n_envs, n_test)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    LiberoImageWrapper(
                        libero_root=libero_root,
                        bddl_file=bddl_file,
                        init_states_file=init_states_file,
                        shape_meta=shape_meta,
                        camera_size=camera_size,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        self.env_fns = [env_fn] * n_envs
        self.env = SyncVectorEnv(self.env_fns)
        self.env_seeds = [test_start_seed + i for i in range(n_test)]
        self.env_init_fn_dills = []
        for i, seed in enumerate(self.env_seeds):
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)
                env.seed(seed)

            self.env_init_fn_dills.append(dill.dumps(init_fn))

        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        all_rewards = [None] * n_inits
        all_video_paths = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            active = end - start
            init_fns = list(self.env_init_fn_dills[start:end])
            init_fns.extend([self.env_init_fn_dills[0]] * (n_envs - active))
            env.call_each(
                "run_dill_function", args_list=[(item,) for item in init_fns]
            )
            obs = env.reset()
            policy.reset()
            done = False
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval LIBERO {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            while not done:
                obs_dict = dict_apply(
                    dict(obs), lambda x: torch.from_numpy(x).to(device=device)
                )
                with torch.no_grad():
                    action = policy.predict_action(obs_dict)["action"]
                action = action.detach().cpu().numpy()
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN or Inf action")
                obs, reward, dones, info = env.step(action)
                done = np.all(dones)
                pbar.update(action.shape[1])
            pbar.close()
            all_rewards[start:end] = env.call("get_attr", "reward")[:active]
            all_video_paths[start:end] = env.render()[:active]

        log_data = {}
        scores = []
        for seed, rewards, video_path in zip(
            self.env_seeds, all_rewards, all_video_paths
        ):
            score = float(np.max(rewards))
            scores.append(score)
            log_data[f"test/sim_max_reward_{seed}"] = score
            if video_path is not None:
                log_data[f"test/sim_video_{seed}"] = wandb.Video(video_path)
        log_data["test/mean_score"] = float(np.mean(scores))
        return log_data
