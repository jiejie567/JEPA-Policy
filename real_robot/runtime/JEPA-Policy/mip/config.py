from dataclasses import dataclass, field


@dataclass
class LogConfig:
    log_dir: str
    model_dir: str | None
    wandb_mode: str
    project: str
    group: str
    exp_name: str
    entity: str | None = "jepa-policy"
    eval_freq: int = 20000
    log_freq: int = 1000
    save_freq: int = 10000
    eval_episodes: int = 10
    save_video: bool = False
    gradient_diagnostic_freq: int = 1000
    validation_freq: int = 10000
    validation_batch_size: int = 16
    validation_seed: int = 12345
    validation_delta_t: float = 1.0
    # Exact optimizer-step counts at which to save lightweight, model-only
    # trajectory checkpoints. Step 0 is the initialized model before updates.
    snapshot_steps: list[int] = field(default_factory=list)


@dataclass
class EvalConfig:
    parallel_rollout: bool = False
    parallel_rollout_workers: int = 2
    persistent_workers: bool = True
    rollout_seed: int = 12345
    episodes_per_worker: int = 2
    worker_timeout_seconds: int = 900
    output_path: str = "/tmp/parallel_image_rollout.json"
    num_steps: int = 1



@dataclass
class OptimizationConfig:
    seed: int = 0
    loss_type: str = "flow"
    loss_scale: float = 100.0
    norm_type: str = "l2"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_steps: int = 1
    sample_mode: str = "stochastic"  # "zero", "mean"
    t_two_step: float = 0.9
    discrete_dt: float = 0.01
    grad_clip_norm: float = 10.0
    ema_rate: float = 0.995
    batch_size: int = 256
    dataloader_num_workers: int = 8
    dataloader_persistent_workers: bool = True
    gradient_steps: int = 150000
    # Optional early termination for trajectory audits. Scheduler horizons still
    # use ``gradient_steps`` so the prefix keeps the formal 300k schedule.
    stop_after_steps: int | None = None
    warmup_ratio: float = 0.0
    rampup_ratio: float = 0.5
    min_value: float = 0.0
    max_value: float = 1.0
    model_path: str | None = None
    interp_type: str = "linear"  # "linear" or "trig"
    device: str = "cuda"
    disable_cudnn: bool = False
    use_compile: bool = True  # Whether to use torch.compile for acceleration
    auto_resume: bool = True  # Whether to automatically resume from checkpoint
    encoder_checkpoint_path: str | None = None
    encoder_checkpoint_use_ema: bool = True
    freeze_encoder: bool = False
    use_future_embed_loss: bool = False
    future_embed_loss_mode: str = "direct"  # "direct" or "mip_two_step"
    future_joint_mode: bool = False
    # Preserve the complete joint forward path while allowing the auxiliary
    # future loss to update only the future projection head. Action loss still
    # updates the encoder, shared transformer, token embeddings, and action head.
    future_head_only_stopgrad: bool = False
    future_t_two_step: float = 0.9
    future_embed_loss_weight: float = 0.01
    future_state_loss_weight: float = 0.1
    future_state_loss_mode: str = "fixed"  # "fixed" or "ratio"
    future_state_loss_ratio: float = 0.05
    future_state_loss_weight_min: float = 1e-4
    future_state_loss_weight_max: float = 1.0
    future_target_type: str = "state"
    use_sigreg: bool = False
    sigreg_weight: float = 0.0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024



@dataclass
class NetworkConfig:
    network_type: str = "mlp"  # "mlp" or "cnn"
    num_layers: int = 4
    emb_dim: int = 512
    dropout: float = 0.1
    encoder_dropout: float = 0.0
    expansion_factor: int = 4
    timestep_emb_dim: int = 128
    timestep_emb_type: str = "positional"  # Type of timestep embedding
    # State encoder configs
    num_encoder_layers: int = 2  # Number of layers for MLP encoder
    # Image encoder configs
    rgb_model_name: str = "resnet18"
    rgb_model_weights: str | None = None
    imagenet_norm: bool = False
    use_seq: bool = True
    keep_horizon_dims: bool = True
    # Transformer specific configs
    n_heads: int = 6
    n_cond_layers: int = 0
    attn_dropout: float = 0.1
    use_causal_mask: bool = False
    use_memory_mask: bool = False
    # Prevent action-token queries from attending to future-token keys while
    # retaining the shared Transformer and the future auxiliary objective.
    block_action_from_future: bool = False
    # UNet specific configs
    model_dim: int = 256
    kernel_size: int = 5
    cond_predict_scale: bool = True
    obs_as_global_cond: bool = True
    dim_mult: list[int] | None = None
    norm_type: str = "groupnorm"
    attention: bool = False
    # RNN specific configs
    rnn_type: str = "LSTM"  # "LSTM" or "GRU"
    max_freq: float = 100.0
    # dinov2 image encoder configs
    encoder_type: str = "image"
    dino_model_name: str = "dinov2_vits14"
    dino_embed_dim: int = 384
    dino_out_dim: int = 768
    n_future_tokens: int = 0
    # Dual ChiTransformer-only settings. Legacy network types ignore these
    # fields so existing configurations and checkpoints retain their behavior.
    future_emb_dim: int = 192
    future_num_layers: int = 8
    future_n_heads: int = 6
    future_ffn_dim: int = 768



@dataclass
class TaskConfig:
    env_name: str = "lift"
    obs_type: str = "state"
    env_type: str = "ph"
    abs_action: bool = True
    # Dataset configuration - either HuggingFace or local path
    dataset_repo: str | None = (
        None  # HuggingFace repository ID (e.g., "ChaoyiPan/mip-dataset")
    )
    dataset_filename: str | None = (
        None  # Path within the repository (e.g., "robomimic/lift/ph/image.hdf5")
    )
    dataset_path: str | None = (
        None  # Local path (deprecated, use dataset_repo/dataset_filename)
    )
    max_episode_steps: int = 400
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
    )
    obs_dim: int = -1
    act_dim: int = 10
    obs_steps: int = 2
    act_steps: int = 8
    horizon: int = 10  # Prediction horizon (typically obs_steps + act_steps)
    num_envs: int = 1
    save_video: bool = False
    shape_meta: dict = field(default_factory=dict)
    render_obs_key: str = "agentview_image"
    val_dataset_percentage: float = 0.0
    # Image observation settings
    rgb_model: str = "resnet18"
    resize_shape: list[int] | None = None
    crop_shape: list[int] | dict[str, list[int]] | None = None
    crop_ratio: float | dict[str, float] | None = None
    random_crop: bool = True
    crop_mode: str | None = None
    eval_crop_mode: str = "center"
    temporal_consistent_crop: bool = False
    use_group_norm: bool = True
    use_seq: bool = True
    libero_benchmark_name: str | None = None
    libero_task_name: str | None = None
    libero_root: str | None = None
    libero_camera_size: int = 128
    robocasa_split: str = "target"
    future_state_enabled: bool = False
    future_state_steps: int = 1
    future_state_steps_list: list[int] = field(default_factory=list)
    future_target_type: str = "embedding"
    use_action_history: bool = False


@dataclass
class Config:
    optimization: OptimizationConfig
    network: NetworkConfig
    task: TaskConfig
    log: LogConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    mode: str = "train"  # "train" or "eval"
