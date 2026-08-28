import torch

import diffusion_policy.model.common.tensor_util as tu
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer


def _crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """DP crop helper with the JEPA-inclusive right/bottom boundary contract."""
    if crop_indices.ndim == images.ndim - 2:
        crop_indices = crop_indices.unsqueeze(-2)
        squeeze_crop_dim = True
    else:
        squeeze_crop_dim = False
    if images.shape[:-3] != crop_indices.shape[:-2]:
        raise ValueError("image and crop-index leading dimensions do not match")

    device = images.device
    channels, image_h, image_w = images.shape[-3:]
    if not (crop_indices[..., 0] >= 0).all().item():
        raise ValueError("negative crop top")
    if not (crop_indices[..., 0] <= image_h - crop_height).all().item():
        raise ValueError("crop top exceeds image boundary")
    if not (crop_indices[..., 1] >= 0).all().item():
        raise ValueError("negative crop left")
    if not (crop_indices[..., 1] <= image_w - crop_width).all().item():
        raise ValueError("crop left exceeds image boundary")

    grid_h = torch.arange(crop_height, device=device)
    grid_h = tu.unsqueeze_expand_at(grid_h, size=crop_width, dim=-1)
    grid_w = torch.arange(crop_width, device=device)
    grid_w = tu.unsqueeze_expand_at(grid_w, size=crop_height, dim=0)
    crop_grid = torch.cat((grid_h.unsqueeze(-1), grid_w.unsqueeze(-1)), dim=-1)
    grid_shape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_indices = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_grid.reshape(
        grid_shape
    )
    all_indices = all_indices[..., 0] * image_w + all_indices[..., 1]
    all_indices = tu.unsqueeze_expand_at(all_indices, size=channels, dim=-3)
    all_indices = tu.flatten(all_indices, begin_axis=-2)

    num_crops = crop_indices.shape[-2]
    images_to_crop = tu.unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = tu.flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_indices)
    last_axis = len(crops.shape) - 1
    crops = tu.reshape_dimensions(
        crops,
        begin_axis=last_axis,
        end_axis=last_axis,
        target_dims=(crop_height, crop_width),
    )
    if squeeze_crop_dim:
        crops = crops.squeeze(-4)
    return crops


class TemporalConsistentCropRandomizer(CropRandomizer):
    """Use one crop offset for every observation frame in each sample."""

    def __init__(self, *args, temporal_group_size: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if temporal_group_size < 1:
            raise ValueError("temporal_group_size must be positive")
        self.temporal_group_size = int(temporal_group_size)

    def forward_in(self, inputs):
        if not self.training or self.temporal_group_size == 1:
            return super().forward_in(inputs)

        group_size = self.temporal_group_size
        if inputs.ndim != 4 or inputs.shape[0] % group_size != 0:
            raise ValueError(
                "expected flattened [batch*time, channels, height, width] input "
                f"with batch dimension divisible by {group_size}; got {tuple(inputs.shape)}"
            )

        batch_size = inputs.shape[0] // group_size
        _, channels, height, width = inputs.shape
        max_top = height - self.crop_height
        max_left = width - self.crop_width
        top = torch.randint(
            0, max_top + 1, (batch_size, self.num_crops, 1), device=inputs.device
        )
        left = torch.randint(
            0, max_left + 1, (batch_size, self.num_crops, 1), device=inputs.device
        )
        crop_indices = torch.cat((top, left), dim=-1)
        crop_indices = crop_indices[:, None].expand(
            batch_size, group_size, self.num_crops, 2
        )
        grouped = inputs.reshape(batch_size, group_size, channels, height, width)
        crops = _crop_image_from_indices(
            grouped,
            crop_indices,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
        )
        return crops.reshape(
            batch_size * group_size * self.num_crops,
            channels,
            self.crop_height,
            self.crop_width,
        )
