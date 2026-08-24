import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

from mip.encoders import (
    BaseEncoder,
    CropRandomizer,
    make_crop_config_record,
    resolve_crop_mode,
    resolve_crop_shape,
    value_for_camera,
)


class FrozenDINOv2Backbone(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14", pretrained: bool = True):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W), float in [0, 1]
        out = self.model(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            if "x_prenorm" in out:
                return out["x_prenorm"][:, 0]
        return out


class FrozenDINOv2ObsEncoder(BaseEncoder):
    def __init__(
        self,
        shape_meta: dict,
        dino_model_name: str = "dinov2_vits14",
        dino_embed_dim: int = 384,
        emb_dim: int = 256,
        resize_shape=None,
        crop_shape=None,
        crop_ratio=None,
        crop_mode: str | None = None,
        eval_crop_mode: str = "center",
        random_crop: bool = True,
        temporal_consistent_crop: bool = False,
        use_seq: bool = False,
        keep_horizon_dims: bool = False,
    ):
        super().__init__()
        rgb_keys = []
        low_dim_keys = []
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}
        crop_config_records = []

        resolved_crop_mode = resolve_crop_mode(
            crop_mode,
            crop_shape,
            crop_ratio,
            random_crop,
            temporal_consistent_crop,
        )
        if resolved_crop_mode == "temporal_consistent":
            raise ValueError(
                "DINOv2 encoder does not support temporal_consistent crop; "
                "use an explicit center/independent mode or no crop"
            )
        eval_crop_mode = str(eval_crop_mode).lower()
        if eval_crop_mode != "center":
            raise ValueError(
                "Only deterministic eval_crop_mode='center' is supported, "
                f"got {eval_crop_mode!r}"
            )

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if obs_type == "rgb":
                rgb_keys.append(key)

                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    h, w = value_for_camera(resize_shape, key)
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                this_randomizer = nn.Identity()
                resolved_crop_shape = None
                if resolved_crop_mode != "none":
                    resolved_crop_shape = resolve_crop_shape(
                        input_shape,
                        crop_shape=crop_shape,
                        crop_ratio=crop_ratio,
                        key=key,
                    )
                    h, w = resolved_crop_shape
                    if resolved_crop_mode == "independent":
                        if h >= input_shape[-2] or w >= input_shape[-1]:
                            raise ValueError(
                                "Random crop must be smaller than its input for "
                                f"{key!r}: input={input_shape[-2:]}, crop={(h, w)}"
                            )
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))
                output_shape = (
                    resolved_crop_shape
                    if resolved_crop_shape is not None
                    else tuple(input_shape[-2:])
                )
                crop_config_records.append(
                    make_crop_config_record(
                        key=key,
                        source_shape=shape,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        crop_mode=resolved_crop_mode,
                        eval_crop_mode=eval_crop_mode,
                    )
                )

                # DINOv2 expects ImageNet normalization
                this_normalizer = torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )

                key_transform_map[key] = nn.Sequential(
                    this_resizer, this_randomizer, this_normalizer
                )
            elif obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_transform_map = key_transform_map
        self.key_shape_map = key_shape_map
        self.shape_meta = shape_meta
        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims
        self.crop_mode = resolved_crop_mode
        self.crop_ratio = crop_ratio
        self.eval_crop_mode = eval_crop_mode
        self.crop_config_records = crop_config_records

        self.rgb_backbone = FrozenDINOv2Backbone(model_name=dino_model_name)

        total_in_dim = len(self.rgb_keys) * dino_embed_dim
        for key in self.low_dim_keys:
            total_in_dim += self.key_shape_map[key][-1]

        self.mlp = nn.Sequential(
            nn.Linear(total_in_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.dino_embed_dim = dino_embed_dim

    def _flatten_if_seq(self, obs_dict):
        if self.use_seq:
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].flatten(end_dim=1)
        return obs_dict

    def encode_rgb_features(self, obs_dict):
        obs_dict = {k: v for k, v in obs_dict.items()}

        features = []

        for key in self.rgb_keys:
            img = obs_dict[key]

            # Support both:
            # (B, T, C, H, W) for sequence inputs
            # (B, C, H, W) for single-frame future targets
            if img.dim() == 5:
                b, t, c, h, w = img.shape
                img = img.reshape(b * t, c, h, w)
                img = self.key_transform_map[key](img)
                feat = self.rgb_backbone(img)
                feat = feat.reshape(b, t, -1)
            elif img.dim() == 4:
                img = self.key_transform_map[key](img)
                feat = self.rgb_backbone(img)
            else:
                raise RuntimeError(
                    f"Unexpected image shape for key {key}: {img.shape}"
                )

            features.append(feat)

        if len(features) == 0:
            raise RuntimeError("FrozenDINOv2ObsEncoder requires at least one rgb key")

        return torch.cat(features, dim=-1)

    def encode_raw_dino(self, obs_dict):
        features = self.encode_rgb_features(obs_dict)
        if features.dim() == 3:
            features = features.mean(dim=1)
        return features

    def rgb_feature_dim(self):
        return len(self.rgb_keys) * self.dino_embed_dim

    def multi_image_forward(self, obs_dict):
        obs_dict = {k: v for k, v in obs_dict.items()}
        obs_dict = self._flatten_if_seq(obs_dict)

        features = []
        batch_size = None

        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            img = self.key_transform_map[key](img)
            feat = self.rgb_backbone(img)
            features.append(feat)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            features.append(data)

        return torch.cat(features, dim=-1)

    def forward(self, obs_dict, mask=None):
        ori_batch_size, ori_seq_len = self.get_batch_size(obs_dict)
        features = self.multi_image_forward(obs_dict)
        result = self.mlp(features)

        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.reshape(ori_batch_size, ori_seq_len, -1)
            else:
                result = result.reshape(ori_batch_size, -1)
        return result

    def get_batch_size(self, obs_dict):
        any_key = next(iter(obs_dict))
        any_tensor = obs_dict[any_key]
        return any_tensor.size(0), any_tensor.size(1)
