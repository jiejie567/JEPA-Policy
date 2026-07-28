from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.encoders import (
    CropRandomizer,
    MultiImageObsEncoder,
    resolve_crop_shape,
    sample_random_image_crops,
)
from mip.networks.chitfm import ChiTransformer


def _shape_meta():
    return {
        "obs": {
            "camera_a": {"shape": [3, 128, 128], "type": "rgb"},
            "camera_b": {"shape": [3, 128, 128], "type": "rgb"},
            "state": {"shape": [9], "type": "low_dim"},
        }
    }


def _encoder(crop: bool, emb_dim: int = 32):
    return MultiImageObsEncoder(
        shape_meta=_shape_meta(),
        rgb_model_name="resnet18",
        rgb_model_weights=None,
        emb_dim=emb_dim,
        resize_shape=None,
        crop_shape=None,
        crop_ratio=0.9 if crop else None,
        crop_mode="temporal_consistent" if crop else "none",
        eval_crop_mode="center",
        random_crop=crop,
        temporal_consistent_crop=crop,
        use_group_norm=True,
        imagenet_norm=False,
        use_seq=True,
        keep_horizon_dims=True,
    )


def _batch(batch_size: int = 2):
    obs = {
        "camera_a": torch.randn(batch_size, 2, 3, 128, 128),
        "camera_b": torch.randn(batch_size, 2, 3, 128, 128),
        "state": torch.randn(batch_size, 2, 9),
    }
    future = {
        "camera_a": torch.randn(batch_size, 3, 128, 128),
        "camera_b": torch.randn(batch_size, 3, 128, 128),
        "state": torch.randn(batch_size, 9),
    }
    return obs, future


def test_no_crop_and_crop116_have_identical_model_output_shapes():
    batch_size = 2
    obs, future = _batch(batch_size)
    action = torch.zeros(batch_size, 16, 7)
    s = torch.zeros(batch_size)
    t = torch.full((batch_size,), 0.9)

    observed_shapes = []
    for use_crop in (False, True):
        torch.manual_seed(5)
        encoder = _encoder(use_crop)
        if use_crop:
            obs_embedding, future_embedding, _ = encoder.encode_obs_and_future_rgb(
                obs, future
            )
        else:
            obs_embedding = encoder(obs)
            future_embedding = encoder.encode_rgb_features(future)

        network = ChiTransformer(
            act_dim=7,
            Ta=16,
            obs_dim=32,
            To=2,
            d_model=32,
            nhead=4,
            num_layers=1,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
            n_cond_layers=0,
            n_future_tokens=1,
            future_out_dim=future_embedding.shape[-1],
        )
        action_pred, future_pred = network.joint_forward(
            action,
            s,
            t,
            obs_embedding,
            torch.zeros_like(future_embedding),
        )
        observed_shapes.append(
            (
                obs_embedding.shape,
                future_embedding.shape,
                action_pred.shape,
                future_pred.shape,
                network.pos_emb.shape,
            )
        )

    assert observed_shapes[0] == observed_shapes[1]
    assert observed_shapes[0][:4] == (
        torch.Size([2, 2, 32]),
        torch.Size([2, 1024]),
        torch.Size([2, 16, 7]),
        torch.Size([2, 1024]),
    )


def test_temporal_crop_reuses_coordinates_across_obs_and_future():
    torch.manual_seed(17)
    encoder = _encoder(True)
    batch_size = 4
    spatial = (
        torch.arange(128)[:, None] * 128 + torch.arange(128)[None, :]
    ).float()
    spatial = spatial[None, None].expand(batch_size, 3, 128, 128)
    obs = {
        "camera_a": torch.stack((spatial, spatial + 100000), dim=1),
        "camera_b": torch.stack((spatial + 300000, spatial + 400000), dim=1),
        "state": torch.zeros(batch_size, 2, 9),
    }
    future = {
        "camera_a": torch.stack((spatial + 200000, spatial + 300000), dim=1),
        "camera_b": torch.stack((spatial + 500000, spatial + 600000), dim=1),
        "state": torch.zeros(batch_size, 2, 9),
    }

    cropped_obs, cropped_future, coordinates = (
        encoder.prepare_temporally_consistent_crops(obs, future)
    )
    assert torch.equal(
        cropped_obs["camera_a"][:, 0],
        cropped_obs["camera_a"][:, 1] - 100000,
    )
    assert torch.equal(
        cropped_obs["camera_a"][:, 0],
        cropped_future["camera_a"][:, 0] - 200000,
    )
    assert torch.equal(
        cropped_obs["camera_a"][:, 0],
        cropped_future["camera_a"][:, 1] - 300000,
    )
    assert torch.equal(
        cropped_obs["camera_b"][:, 0],
        cropped_obs["camera_b"][:, 1] - 100000,
    )
    assert torch.equal(
        cropped_obs["camera_b"][:, 0],
        cropped_future["camera_b"][:, 0] - 200000,
    )
    assert torch.equal(
        cropped_obs["camera_b"][:, 0],
        cropped_future["camera_b"][:, 1] - 300000,
    )
    assert coordinates["camera_a"].shape == (batch_size, 2)
    assert coordinates["camera_b"].shape == (batch_size, 2)
    # Cameras sample independently; a fixed seed makes this non-probabilistic.
    assert not torch.equal(coordinates["camera_a"], coordinates["camera_b"])


def test_eval_center_crop_is_deterministic_and_does_not_consume_rng():
    cropper = CropRandomizer((3, 128, 128), 116, 116)
    cropper.eval()
    images = torch.randn(3, 3, 3, 128, 128)
    rng_before = torch.get_rng_state().clone()
    first, first_coordinates = cropper.forward_temporally_consistent(images)
    rng_after_first = torch.get_rng_state().clone()
    second, second_coordinates = cropper.forward_temporally_consistent(images)
    rng_after_second = torch.get_rng_state().clone()

    assert torch.equal(first, second)
    assert torch.equal(first_coordinates, second_coordinates)
    assert torch.equal(first_coordinates, torch.full((3, 2), 6))
    assert torch.equal(rng_before, rng_after_first)
    assert torch.equal(rng_before, rng_after_second)


def test_crop_sizes_use_the_same_approximately_90_percent_edge_definition():
    cases = (
        (84, 76, 4, 76 / 84),
        (128, 116, 6, 116 / 128),
    )
    for input_size, crop_size, center_offset, edge_retention in cases:
        assert 0.90 <= edge_retention < 0.91
        assert resolve_crop_shape(
            (3, input_size, input_size),
            crop_ratio=0.9,
            key="camera",
        ) == (crop_size, crop_size)

        cropper = CropRandomizer(
            (3, input_size, input_size), crop_size, crop_size
        )
        cropper.eval()
        image = torch.arange(input_size * input_size).reshape(
            1, 1, input_size, input_size
        )
        cropped, coordinates = cropper.forward_temporally_consistent(image)
        assert cropped.shape[-2:] == (crop_size, crop_size)
        assert torch.equal(
            coordinates, torch.tensor([[center_offset, center_offset]])
        )

        # The full random-crop support includes both the top-left and the final
        # valid bottom-right window.
        bottom_right = torch.tensor(
            [[input_size - crop_size, input_size - crop_size]]
        )
        edge_crop, used_coordinates = cropper.forward_temporally_consistent(
            image, crop_indices=bottom_right
        )
        assert edge_crop.shape[-2:] == (crop_size, crop_size)
        assert torch.equal(used_coordinates, bottom_right)


def test_ratio_crop_resolves_rectangular_and_per_camera_shapes():
    assert resolve_crop_shape(
        (3, 96, 160), crop_ratio=0.9, key="camera"
    ) == (86, 144)
    assert resolve_crop_shape(
        (3, 128, 128),
        crop_ratio={"agent": 0.9, "wrist": 0.8},
        key="wrist",
    ) == (102, 102)


def test_explicit_crop_shape_overrides_ratio():
    assert resolve_crop_shape(
        (3, 128, 128),
        crop_shape=(112, 110),
        crop_ratio=0.9,
        key="camera",
    ) == (112, 110)


def test_independent_random_crop_can_sample_exact_bottom_right():
    image = torch.arange(128 * 128).reshape(1, 1, 128, 128)

    def final_valid_index(high, size, device=None, **_kwargs):
        return torch.full(size, high - 1, device=device, dtype=torch.long)

    with patch("mip.encoders.torch.randint", side_effect=final_valid_index):
        cropped, coordinates = sample_random_image_crops(
            image,
            crop_height=116,
            crop_width=116,
            num_crops=1,
        )

    assert torch.equal(coordinates, torch.tensor([[[12, 12]]]))
    assert torch.equal(cropped[0, 0], image[0, :, 12:, 12:])


def test_no_crop_transform_is_exact_identity():
    encoder = _encoder(False)
    images = torch.randn(4, 3, 128, 128)
    for key in encoder.rgb_keys:
        transformed = encoder.key_transform_map[key](images)
        assert transformed.shape == images.shape
        assert torch.equal(transformed, images)


def test_future_target_path_samples_one_crop_per_camera_only_once():
    torch.manual_seed(23)
    encoder = _encoder(True)
    obs, future = _batch(batch_size=2)
    croppers = {
        key: encoder.key_transform_map[key][1] for key in encoder.rgb_keys
    }

    with (
        patch.object(
            croppers["camera_a"],
            "forward_temporally_consistent",
            wraps=croppers["camera_a"].forward_temporally_consistent,
        ) as camera_a_crop,
        patch.object(
            croppers["camera_b"],
            "forward_temporally_consistent",
            wraps=croppers["camera_b"].forward_temporally_consistent,
        ) as camera_b_crop,
    ):
        obs_embedding, future_embedding, coordinates = (
            encoder.encode_obs_and_future_rgb(obs, future)
        )

    assert camera_a_crop.call_count == 1
    assert camera_b_crop.call_count == 1
    assert camera_a_crop.call_args.args[0].shape[1] == 3
    assert camera_b_crop.call_args.args[0].shape[1] == 3
    assert obs_embedding.shape == (2, 2, 32)
    assert future_embedding.shape == (2, 1024)
    assert set(coordinates) == set(encoder.rgb_keys)
    assert all(
        torch.equal(coordinates[key].cpu(), encoder.last_crop_params[key])
        for key in encoder.rgb_keys
    )


def test_four_crop_ablation_configs_only_vary_future_crop_and_run_name():
    config_dir = str((Path(__file__).parents[1] / "examples" / "configs").resolve())
    names = [
        "exps/mug_mug_baseline_nocrop_seed42",
        "exps/mug_mug_baseline_crop116_temporal_seed42",
        "exps/mug_mug_future4_ratio010_nocrop_seed42",
        "exps/mug_mug_future4_ratio010_crop116_temporal_seed42",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        configs = [compose(config_name=name) for name in names]

    for config in configs:
        assert config.optimization.seed == 42
        assert config.optimization.batch_size == 256
        assert config.optimization.gradient_steps == 300000
        assert config.optimization.t_two_step == 0.9
        assert config.task.obs_steps == 2
        assert config.task.horizon == 16
        assert config.task.act_steps == 8
        assert config.task.future_state_steps == 4
        assert config.optimization.future_state_loss_ratio == 0.1
        assert config.log.eval_freq == 10000
        assert config.log.eval_episodes == 40
        assert config.eval.parallel_rollout_workers == 20
        assert config.eval.rollout_seed == 12345

    allowed_differences = {
        "optimization.use_future_embed_loss",
        "optimization.future_joint_mode",
        "network.n_future_tokens",
        "task.future_state_enabled",
        "task.crop_shape",
        "task.crop_ratio",
        "task.random_crop",
        "task.crop_mode",
        "task.temporal_consistent_crop",
        "log.exp_name",
        "log.log_dir",
    }

    def flattened(config):
        result = {}

        def visit(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(f"{prefix}.{key}" if prefix else key, item)
            else:
                result[prefix] = value

        visit("", OmegaConf.to_container(config, resolve=True))
        return {
            key: value for key, value in result.items() if key not in allowed_differences
        }

    reference = flattened(configs[0])
    assert all(flattened(config) == reference for config in configs[1:])

    assert configs[0].task.crop_shape is None
    assert configs[2].task.crop_shape is None
    assert configs[1].task.crop_shape is None
    assert configs[3].task.crop_shape is None
    assert configs[0].task.crop_ratio is None
    assert configs[2].task.crop_ratio is None
    assert configs[1].task.crop_ratio == 0.9
    assert configs[3].task.crop_ratio == 0.9
    assert resolve_crop_shape(
        tuple(configs[1].task.shape_meta.obs.agentview_rgb.shape),
        crop_shape=configs[1].task.crop_shape,
        crop_ratio=configs[1].task.crop_ratio,
        key="agentview_rgb",
    ) == (116, 116)
    assert configs[1].task.temporal_consistent_crop is True
    assert configs[3].task.temporal_consistent_crop is True


def test_dataset_task_configs_resolve_expected_90_percent_crops():
    config_dir = str((Path(__file__).parents[1] / "examples" / "configs").resolve())
    names = [
        "task/mug_mug_image",
        "task/moka_moka_image",
        "task/lift_ph_image",
        "task/can_ph_image",
        "task/square_ph_image",
        "task/tool_hang_ph_image",
        "task/transport_ph_image",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        configs = [compose(config_name=name) for name in names]

    for config in configs:
        task = config.task
        assert task.crop_shape is None
        assert task.crop_ratio == 0.9
        assert task.crop_mode == "temporal_consistent"
        assert task.eval_crop_mode == "center"
        assert task.random_crop is True
        assert task.temporal_consistent_crop is True
        for key, attr in task.shape_meta.obs.items():
            if attr.get("type", "low_dim") != "rgb":
                continue
            input_size = int(attr.shape[-1])
            expected_size = 116 if input_size == 128 else 76
            assert resolve_crop_shape(
                tuple(attr.shape),
                crop_shape=task.crop_shape,
                crop_ratio=task.crop_ratio,
                key=key,
            ) == (expected_size, expected_size)


def test_dino_task_keeps_explicit_model_specific_crop_override():
    config_dir = str((Path(__file__).parents[1] / "examples" / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="task/tool_hang_ph_image_dino")

    task = config.task
    assert task.crop_shape == [84, 84]
    assert task.crop_ratio is None
    assert task.crop_mode == "center"
    assert task.eval_crop_mode == "center"
    assert task.random_crop is False
    assert task.temporal_consistent_crop is False
