"""Observation encoder.

Port from https://github.com/CleanDiffuserTeam/CleanDiffuser

Author: Chaoyi Pan
Date: 2025-10-03
"""

import copy
from collections.abc import Callable, Mapping

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as ttf

import jepa_policy.torch_utils as tu
from jepa_policy.torch_utils import at_least_ndim


VALID_CROP_MODES = {"none", "center", "independent", "temporal_consistent"}


def resolve_crop_mode(
    crop_mode: str | None,
    crop_shape,
    crop_ratio,
    random_crop: bool,
    temporal_consistent_crop: bool,
) -> str:
    """Resolve the authoritative crop mode while preserving legacy callers."""
    has_crop_spec = crop_shape is not None or crop_ratio is not None
    if crop_mode is None:
        if not has_crop_spec:
            return "none"
        if temporal_consistent_crop:
            return "temporal_consistent"
        return "independent" if random_crop else "center"

    mode = str(crop_mode).lower()
    if mode not in VALID_CROP_MODES:
        raise ValueError(
            f"Unsupported crop_mode={crop_mode!r}; expected one of "
            f"{sorted(VALID_CROP_MODES)}"
        )
    if mode != "none" and not has_crop_spec:
        raise ValueError(
            f"crop_mode={mode!r} requires crop_shape or crop_ratio"
        )

    expected_random = mode in {"independent", "temporal_consistent"}
    expected_temporal = mode == "temporal_consistent"
    if bool(random_crop) != expected_random:
        raise ValueError(
            f"crop_mode={mode!r} requires random_crop={expected_random}, "
            f"got {random_crop}"
        )
    if bool(temporal_consistent_crop) != expected_temporal:
        raise ValueError(
            f"crop_mode={mode!r} requires "
            f"temporal_consistent_crop={expected_temporal}, "
            f"got {temporal_consistent_crop}"
        )
    return mode


def value_for_camera(value, key: str):
    if isinstance(value, Mapping):
        if key not in value:
            raise ValueError(f"Missing crop setting for RGB key {key!r}")
        return value[key]
    return value


def resolve_crop_shape(
    input_shape: tuple[int, int, int],
    *,
    crop_shape=None,
    crop_ratio=None,
    key: str,
) -> tuple[int, int] | None:
    """Resolve an explicit or ratio-based crop for one camera.

    Explicit shapes take precedence. Ratio crops remove a symmetric margin from
    each edge, which maps 84 -> 76 and 128 -> 116 for ``crop_ratio=0.9``.
    """
    if len(input_shape) != 3:
        raise ValueError(f"Expected CHW input shape, got {input_shape}")
    image_h, image_w = (int(input_shape[-2]), int(input_shape[-1]))

    explicit_shape = value_for_camera(crop_shape, key)
    if explicit_shape is not None:
        if len(explicit_shape) != 2:
            raise ValueError(
                f"crop_shape for {key!r} must contain [height, width], "
                f"got {explicit_shape}"
            )
        crop_h, crop_w = (int(explicit_shape[0]), int(explicit_shape[1]))
    else:
        ratio_value = value_for_camera(crop_ratio, key)
        if ratio_value is None:
            return None
        ratio = float(ratio_value)
        if not 0.0 < ratio < 1.0:
            raise ValueError(
                f"crop_ratio for {key!r} must be between 0 and 1, got {ratio}"
            )
        margin_h = max(1, round(image_h * (1.0 - ratio) / 2.0))
        margin_w = max(1, round(image_w * (1.0 - ratio) / 2.0))
        crop_h = image_h - 2 * margin_h
        crop_w = image_w - 2 * margin_w

    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(
            f"Resolved non-positive crop for {key!r}: ({crop_h}, {crop_w})"
        )
    if crop_h > image_h or crop_w > image_w:
        raise ValueError(
            f"Crop for {key!r} exceeds input: input=({image_h}, {image_w}), "
            f"crop=({crop_h}, {crop_w})"
        )
    return crop_h, crop_w


def make_crop_config_record(
    *,
    key: str,
    source_shape: tuple[int, int, int],
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int],
    crop_mode: str,
    eval_crop_mode: str,
) -> dict:
    input_h, input_w = input_shape[-2:]
    output_h, output_w = output_shape
    max_top = input_h - output_h
    max_left = input_w - output_w
    active = crop_mode != "none"
    return {
        "camera": key,
        "source_hw": tuple(source_shape[-2:]),
        "input_hw": (input_h, input_w),
        "output_hw": (output_h, output_w),
        "retained_hw": (output_h / input_h, output_w / input_w),
        "train_mode": crop_mode,
        "eval_mode": eval_crop_mode if active else "none",
        "train_offsets": (
            ((0, max_top), (0, max_left))
            if crop_mode in {"independent", "temporal_consistent"}
            else None
        ),
        "eval_offset": ((max_top // 2, max_left // 2) if active else None),
    }


def format_crop_config_record(record: dict) -> str:
    """Format a resolved per-camera crop record for startup logs."""
    source_h, source_w = record["source_hw"]
    input_h, input_w = record["input_hw"]
    output_h, output_w = record["output_hw"]
    retained_h, retained_w = record["retained_hw"]
    message = (
        f"Crop config: camera={record['camera']} source={source_h}x{source_w} "
        f"input={input_h}x{input_w} output={output_h}x{output_w} "
        f"retained={retained_h:.3%}x{retained_w:.3%} "
        f"train={record['train_mode']} eval={record['eval_mode']}"
    )
    if record["train_offsets"] is not None:
        (top_min, top_max), (left_min, left_max) = record["train_offsets"]
        message += (
            f" train_offsets=top[{top_min},{top_max}],left[{left_min},{left_max}]"
        )
    if record["eval_offset"] is not None:
        message += f" eval_offset={record['eval_offset']}"
    return message


def get_mask(
    mask: torch.Tensor,
    mask_shape: tuple,
    dropout: float,
    train: bool,
    device: torch.device,
):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1.0 if mask is None else mask
    return mask


class CropRandomizer(nn.Module):
    """Randomly sample crops at input, and then average across crop features at output."""

    def __init__(
        self,
        input_shape,
        crop_height,
        crop_width,
        num_crops=1,
        pos_enc=False,
    ):
        """Args:
        input_shape (tuple, list): shape of input (not including batch dimension)
        crop_height (int): crop height
        crop_width (int): crop width
        num_crops (int): number of random crops to take
        pos_enc (bool): if True, add 2 channels to the output to encode the spatial
            location of the cropped pixels in the source image.
        """
        super().__init__()

        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3  # must have at least (C, H, W) dimensions
        if self.training:
            # generate random crops
            out, _ = sample_random_image_crops(
                images=inputs,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                num_crops=self.num_crops,
                pos_enc=self.pos_enc,
            )
            # [B, N, ...] -> [B * N, ...]
            return tu.join_dimensions(out, 0, 1)
        else:
            # take center crop during eval
            out = ttf.center_crop(
                img=inputs, output_size=(self.crop_height, self.crop_width)
            )
            if self.num_crops > 1:
                B, C, H, W = out.shape
                out = (
                    out.unsqueeze(1)
                    .expand(B, self.num_crops, C, H, W)
                    .reshape(-1, C, H, W)
                )
                # [B * N, ...]
            return out

    def forward_out(self, inputs):
        """Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        if self.num_crops <= 1:
            return inputs
        else:
            batch_size = inputs.shape[0] // self.num_crops
            out = tu.reshape_dimensions(
                inputs,
                begin_axis=0,
                end_axis=0,
                target_dims=(batch_size, self.num_crops),
            )
            return out.mean(dim=1)

    def forward(self, inputs):
        return self.forward_in(inputs)

    def forward_temporally_consistent(
        self,
        inputs: torch.Tensor,
        crop_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop every frame of a sample with one shared spatial window.

        Args:
            inputs: ``(B, T, C, H, W)`` or ``(B, C, H, W)`` images.
            crop_indices: Optional ``(B, 2)`` tensor of ``(top, left)``
                coordinates. Supplying coordinates never consumes RNG.

        Returns:
            The cropped images with the same rank as ``inputs`` and the
            per-sample ``(B, 2)`` crop coordinates. Different callers can
            reuse the returned coordinates for additional frames from the
            same camera.
        """
        if self.num_crops != 1:
            raise ValueError(
                "Temporal-consistent crop currently requires num_crops=1"
            )
        if inputs.dim() not in (4, 5):
            raise ValueError(
                "Temporal-consistent crop expects BCHW or BTCHW input, "
                f"got {tuple(inputs.shape)}"
            )

        squeeze_time = inputs.dim() == 4
        sequence = inputs.unsqueeze(1) if squeeze_time else inputs
        batch_size, time_steps = sequence.shape[:2]
        image_h, image_w = sequence.shape[-2:]
        max_sample_h = image_h - self.crop_height
        max_sample_w = image_w - self.crop_width
        if max_sample_h <= 0 or max_sample_w <= 0:
            raise ValueError(
                "Crop must be smaller than its input: "
                f"input=({image_h}, {image_w}), "
                f"crop=({self.crop_height}, {self.crop_width})"
            )

        if crop_indices is None:
            if self.training:
                crop_indices = sample_crop_indices(
                    sequence[:, 0],
                    crop_height=self.crop_height,
                    crop_width=self.crop_width,
                    num_crops=1,
                )[:, 0]
            else:
                # Center crop is deterministic and deliberately avoids RNG.
                crop_h = torch.full(
                    (batch_size,),
                    max_sample_h // 2,
                    device=sequence.device,
                    dtype=torch.long,
                )
                crop_w = torch.full(
                    (batch_size,),
                    max_sample_w // 2,
                    device=sequence.device,
                    dtype=torch.long,
                )
                crop_indices = torch.stack((crop_h, crop_w), dim=-1)
        else:
            crop_indices = crop_indices.to(device=sequence.device, dtype=torch.long)
            if crop_indices.shape != (batch_size, 2):
                raise ValueError(
                    "crop_indices must have shape (B, 2), "
                    f"got {tuple(crop_indices.shape)} for batch {batch_size}"
                )

        temporal_indices = crop_indices[:, None, :].expand(
            batch_size, time_steps, 2
        )
        cropped = crop_image_from_indices(
            images=sequence,
            crop_indices=temporal_indices,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
        )
        if squeeze_time:
            cropped = cropped[:, 0]
        return cropped, crop_indices

    def __repr__(self):
        """Pretty print network."""
        header = f"{str(self.__class__.__name__)}"
        msg = (
            header
            + f"(input_shape={self.input_shape}, crop_size=[{self.crop_height}, {self.crop_width}], num_crops={self.num_crops})"
        )
        return msg


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """Crops images at the locations specified by @crop_indices. Crops will be
    taken across all channels.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_indices (torch.Tensor): batch of indices of shape [..., N, 2] where
            N is the number of crops to take per image and each entry corresponds
            to the pixel height and width of where to take the crop. Note that
            the indices can also be of shape [..., 2] if only 1 crop should
            be taken per image. Leading dimensions must be consistent with
            @images argument. Each index specifies the top left of the crop.
            Values must be in range [0, H - CH] x [0, W - CW] where
            H and W are the height and width of @images and CH and CW are
            @crop_height and @crop_width.

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

    Returns:
        crops (torch.Tesnor): cropped images of shape [..., C, @crop_height, @crop_width]
    """
    # make sure length of input shapes is consistent
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (
        ndim_im_shape == ndim_indices_shape + 2
    )

    # maybe pad so that @crop_indices is shape [..., N, 2]
    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    # make sure leading dimensions between images and indices are consistent
    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    # make sure @crop_indices are in valid range
    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] <= (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] <= (image_w - crop_width)).all().item()

    # convert each crop index (ch, cw) into a list of pixel indices that correspond to the entire window.

    # 2D index array with columns [0, 1, ..., CH - 1] and shape [CH, CW]
    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = tu.unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    # 2D index array with rows [0, 1, ..., CW - 1] and shape [CH, CW]
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = tu.unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    # combine into shape [CH, CW, 2]
    crop_in_grid = torch.cat(
        (crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1
    )

    # Add above grid with the offset index of each sampled crop to get 2d indices for each crop.
    # After broadcasting, this will be shape [..., N, CH, CW, 2] and each crop has a [CH, CW, 2]
    # shape array that tells us which pixels from the corresponding source image to grab.
    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(
        grid_reshape
    )

    # For using @torch.gather, convert to flat indices from 2D indices, and also
    # repeat across the channel dimension. To get flat index of each pixel to grab for
    # each sampled crop, we just use the mapping: ind = h_ind * @image_w + w_ind
    all_crop_inds = (
        all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1]
    )  # shape [..., N, CH, CW]
    all_crop_inds = tu.unsqueeze_expand_at(
        all_crop_inds, size=image_c, dim=-3
    )  # shape [..., N, C, CH, CW]
    all_crop_inds = tu.flatten(
        all_crop_inds, begin_axis=-2
    )  # shape [..., N, C, CH * CW]

    # Repeat and flatten the source images -> [..., N, C, H * W] and then use gather to index with crop pixel inds
    images_to_crop = tu.unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = tu.flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    # [..., N, C, CH * CW] -> [..., N, C, CH, CW]
    reshape_axis = len(crops.shape) - 1
    crops = tu.reshape_dimensions(
        crops,
        begin_axis=reshape_axis,
        end_axis=reshape_axis,
        target_dims=(crop_height, crop_width),
    )

    if is_padded:
        # undo padding -> [..., C, CH, CW]
        crops = crops.squeeze(-4)
    return crops


def sample_crop_indices(images, crop_height, crop_width, num_crops):
    """Uniformly sample every valid top-left crop coordinate, inclusively."""
    image_h, image_w = images.shape[-2:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width
    if max_sample_h < 0 or max_sample_w < 0:
        raise ValueError(
            f"Crop exceeds input: input=({image_h}, {image_w}), "
            f"crop=({crop_height}, {crop_width})"
        )
    sample_shape = (*images.shape[:-3], num_crops)
    crop_inds_h = torch.randint(
        max_sample_h + 1,
        sample_shape,
        device=images.device,
    )
    crop_inds_w = torch.randint(
        max_sample_w + 1,
        sample_shape,
        device=images.device,
    )
    return torch.stack((crop_inds_h, crop_inds_w), dim=-1)


def sample_random_image_crops(
    images, crop_height, crop_width, num_crops, pos_enc=False
):
    """For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
    @images.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

        num_crops (n): number of crops to sample

        pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial
            encoding of the original source pixel locations. This means that the
            output crops will contain information about where in the source image
            it was sampled from.

    Returns:
        crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width)
            if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)

        crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
    """
    device = images.device

    # maybe add 2 channels of spatial encoding to the source image
    source_im = images
    if pos_enc:
        # spatial encoding [y, x] in [0, 1]
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x))  # shape [C, H, W]

        # unsqueeze and expand to match leading dimensions -> shape [..., C, H, W]
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)

        # concat across channel dimension with input
        source_im = torch.cat((source_im, position_enc), dim=-3)

    # Sample crop locations for all tensor dimensions up to the last 3, which are [C, H, W].
    # Each gets @num_crops samples - typically this will just be the batch dimension (B), so
    # we will sample [B, N] indices, but this supports having more than one leading dimension,
    # or possibly no leading dimension.
    crop_inds = sample_crop_indices(
        source_im,
        crop_height=crop_height,
        crop_width=crop_width,
        num_crops=num_crops,
    )

    crops = crop_image_from_indices(
        images=source_im,
        crop_indices=crop_inds,
        crop_height=crop_height,
        crop_width=crop_width,
    )

    return crops, crop_inds


class BaseEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        raise NotImplementedError


class IdentityEncoder(BaseEncoder):
    """Identity encoder does not change the input condition.

    Input:
        - condition: (b, *cond_in_shape) or dict of tensors
        - mask :     (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_in_shape)
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout

    def forward(self, condition: torch.Tensor | dict, mask: torch.Tensor = None):
        # Handle dict input by concatenating all tensors
        if isinstance(condition, dict):
            # Sort keys for consistent ordering and concatenate all values
            keys = sorted(condition.keys())
            tensors = [condition[k] for k in keys]
            # Flatten each tensor to (batch, -1) and concatenate
            flattened = [t.reshape(t.shape[0], -1) for t in tensors]
            condition = torch.cat(flattened, dim=-1)

        mask = at_least_ndim(
            get_mask(
                mask,
                (condition.shape[0],),
                self.dropout,
                self.training,
                condition.device,
            ),
            condition.dim(),
        )
        return condition * mask


class Mlp(nn.Module):
    """**Multilayer perceptron.** A simple pytorch MLP module.

    Args:
        in_dim: int,
            The dimension of the input tensor.
        hidden_dims: List[int],
            A list of integers, each element is the dimension of the hidden layer.
        out_dim: int,
            The dimension of the output tensor.
        activation: nn.Module,
            The activation function used in the hidden layers.
        out_activation: nn.Module,
            The activation function used in the output layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        out_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                    activation,
                )
                for i in range(len(hidden_dims))
            ],
            nn.Linear(hidden_dims[-1], out_dim),
            out_activation,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class MLPEncoder(BaseEncoder):
    """A simple MLP encoder.

    Use a simple MLP to project the input condition to the desired dimension.

    Args:
        in_dim: int,
            The input dimension of the condition
        out_dim: int,
            The output dimension of the condition
        hidden_dims: List[int],
            The hidden dimensions of the MLP
        act: nn.Module,
            The activation function of the MLP
        dropout: float,
            The label dropout rate

    Examples:
        >>> encoder = MLPEncoder(in_dim=5, out_dim=10)
        >>> condition = torch.randn(2, 5)
        >>> encoder(condition).shape
        torch.Size([2, 10])
        >>> condition = torch.randn(2, 20, 5)
        >>> encoder(condition).shape
        torch.Size([2, 20, 10])
    """

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        To: int,
        hidden_dims: list[int],
        act=nn.LeakyReLU(),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dropout = dropout
        self.To = To
        self.emb_dim = emb_dim
        hidden_dims = (
            [
                hidden_dims,
            ]
            if isinstance(hidden_dims, int)
            else hidden_dims
        )
        self.mlp = Mlp(obs_dim * To, hidden_dims, emb_dim * To, act)

    def forward(self, obs: torch.Tensor | dict, mask: torch.Tensor = None):
        # Handle dict input by concatenating all tensors
        if isinstance(obs, dict):
            # Sort keys for consistent ordering and concatenate all values
            keys = sorted(obs.keys())
            obs_list = [obs[k] for k in keys]
        else:
            obs_list = [obs]
        # Flatten each tensor to (batch, -1) and concatenate
        flattened = [t.reshape(t.shape[0], -1) for t in obs_list]
        obs = torch.cat(flattened, dim=-1)

        mask = at_least_ndim(
            get_mask(
                mask,
                (obs.shape[0],),
                self.dropout,
                self.training,
                obs.device,
            ),
            obs.dim(),
        )
        emb_features = self.mlp(obs) * mask
        return emb_features.reshape(obs.shape[0], self.To, self.emb_dim)


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


class ImageNetNormalizer(nn.Module):
    """Map robomimic RGB from [-1, 1] to ImageNet-normalized NCHW."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "ImageNet normalization expects NCHW RGB input, "
                f"got shape {tuple(x.shape)}"
            )
        x = (x + 1.0) / 2.0
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


def resolve_resnet_weights(name: str, weights: str | None):
    if weights is None or str(weights).lower() in {"none", "null"}:
        return None
    if weights != "IMAGENET1K_V1":
        raise ValueError(
            f"Unsupported weights {weights!r} for {name}; expected null or IMAGENET1K_V1"
        )
    weights_enum = {
        "resnet18": torchvision.models.ResNet18_Weights,
        "resnet50": torchvision.models.ResNet50_Weights,
    }.get(name)
    if weights_enum is None:
        raise ValueError(
            f"IMAGENET1K_V1 is only supported for resnet18/resnet50, got {name!r}"
        )
    return weights_enum.IMAGENET1K_V1


def get_resnet(name, weights=None, **kwargs):
    """name: resnet18, resnet34, resnet50
    weights: None or "IMAGENET1K_V1".
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=resolve_resnet_weights(name, weights), **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


class MultiImageObsEncoder(BaseEncoder):
    """Input:
        - condition: {"cond1": (b, *cond1_shape), "cond2": (b, *cond2_shape), ...} or (b, *cond_in_shape)
        - mask :     (b, *mask_shape) or None, None means no mask.

    Output:
        - condition: (b, *cond_out_shape)

    Assumes rgb input: B, C, H, W or B, seq_len, C,H,W
    Assumes low_dim input: B, D or B, seq_len, D
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str,
        rgb_model_weights: str | None = None,
        emb_dim: int = 256,
        resize_shape: tuple[int, int] | dict[str, tuple] | None = None,
        crop_shape: tuple[int, int] | dict[str, tuple] | None = None,
        crop_ratio: float | dict[str, float] | None = None,
        crop_mode: str | None = None,
        eval_crop_mode: str = "center",
        random_crop: bool = True,
        temporal_consistent_crop: bool = False,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
        # use_seq: B, seq_len, C, H, W or B, C, H, W
        use_seq=False,
        # if True: (bs, seq_len, embed_dim)
        keep_horizon_dims=False,
    ):
        super().__init__()
        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
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
        eval_crop_mode = str(eval_crop_mode).lower()
        if eval_crop_mode != "center":
            raise ValueError(
                "Only deterministic eval_crop_mode='center' is supported, "
                f"got {eval_crop_mode!r}"
            )

        # rgb_model
        if "resnet" in rgb_model_name:
            rgb_model = get_resnet(rgb_model_name, weights=rgb_model_weights)
        else:
            raise ValueError("Fatal rgb_model")

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            # print(key, attr)
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    h, w = value_for_camera(resize_shape, key)
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # Resolve the crop after resize so ratios always describe the
                # actual tensor entering the crop transform.
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
                    if resolved_crop_mode in {
                        "independent",
                        "temporal_consistent",
                    }:
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
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = ImageNetNormalizer()

                this_transform = nn.Sequential(
                    this_resizer, this_randomizer, this_normalizer
                )
                key_transform_map[key] = this_transform
            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.crop_mode = resolved_crop_mode
        self.crop_ratio = crop_ratio
        self.eval_crop_mode = eval_crop_mode
        self.random_crop = resolved_crop_mode in {
            "independent",
            "temporal_consistent",
        }
        self.temporal_consistent_crop = (
            resolved_crop_mode == "temporal_consistent"
        )
        self.crop_config_records = crop_config_records
        self._last_crop_params: dict[str, torch.Tensor] = {}

        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims
        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape(), emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def _apply_temporal_rgb_transform(
        self,
        key: str,
        images: torch.Tensor,
        crop_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply resize/crop/normalization while preserving the time axis."""
        if images.dim() not in (4, 5):
            raise ValueError(
                f"RGB key {key} must be BCHW or BTCHW, got {tuple(images.shape)}"
            )
        squeeze_time = images.dim() == 4
        sequence = images.unsqueeze(1) if squeeze_time else images
        batch_size, time_steps, channels = sequence.shape[:3]

        # key_transform_map is built as resize -> crop -> normalize. Apply the
        # rank-sensitive transforms to a flattened copy, but sample the crop on
        # BTCHW so one coordinate is shared across all T frames.
        resizer, randomizer, normalizer = self.key_transform_map[key]
        flat = sequence.reshape(
            batch_size * time_steps, channels, *sequence.shape[-2:]
        )
        flat = resizer(flat)
        sequence = flat.reshape(
            batch_size, time_steps, channels, *flat.shape[-2:]
        )
        if not isinstance(randomizer, CropRandomizer):
            raise RuntimeError(
                "Temporal-consistent crop requires CropRandomizer, "
                f"got {type(randomizer).__name__} for key {key}"
            )
        sequence, crop_indices = randomizer.forward_temporally_consistent(
            sequence, crop_indices=crop_indices
        )
        flat = sequence.reshape(
            batch_size * time_steps, channels, *sequence.shape[-2:]
        )
        flat = normalizer(flat)
        sequence = flat.reshape(
            batch_size, time_steps, channels, *flat.shape[-2:]
        )
        if squeeze_time:
            sequence = sequence[:, 0]
        return sequence, crop_indices

    def prepare_temporally_consistent_crops(
        self,
        obs_dict: dict[str, torch.Tensor],
        future_obs_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor] | None,
        dict[str, torch.Tensor],
    ]:
        """Prepare current and future RGB frames using one crop per camera.

        Current frames and all future frames are concatenated along time before
        cropping. Consequently the future target path cannot independently
        resample a spatial window.
        """
        if not self.temporal_consistent_crop:
            raise RuntimeError(
                "prepare_temporally_consistent_crops requires "
                "temporal_consistent_crop=true"
            )

        prepared_obs = dict(obs_dict)
        prepared_future = (
            None if future_obs_dict is None else dict(future_obs_dict)
        )
        crop_params: dict[str, torch.Tensor] = {}

        for key in self.rgb_keys:
            obs_images = obs_dict[key]
            if obs_images.dim() != 5:
                raise ValueError(
                    "Temporal policy observations must be BTCHW, "
                    f"got {tuple(obs_images.shape)} for key {key}"
                )
            obs_steps = obs_images.shape[1]
            combined = obs_images
            future_was_sequence = False
            if future_obs_dict is not None:
                future_images = future_obs_dict[key]
                if future_images.dim() == 4:
                    future_sequence = future_images.unsqueeze(1)
                elif future_images.dim() == 5:
                    future_sequence = future_images
                    future_was_sequence = True
                else:
                    raise ValueError(
                        "Future RGB observations must be BCHW or BTCHW, "
                        f"got {tuple(future_images.shape)} for key {key}"
                    )
                combined = torch.cat((obs_images, future_sequence), dim=1)

            prepared, indices = self._apply_temporal_rgb_transform(key, combined)
            prepared_obs[key] = prepared[:, :obs_steps]
            if prepared_future is not None:
                future_part = prepared[:, obs_steps:]
                prepared_future[key] = (
                    future_part if future_was_sequence else future_part[:, 0]
                )
            crop_params[key] = indices

        self._last_crop_params = {
            key: value.detach().cpu().clone() for key, value in crop_params.items()
        }
        return prepared_obs, prepared_future, crop_params

    @property
    def last_crop_params(self) -> dict[str, torch.Tensor]:
        """Most recent per-camera ``(top, left)`` values, for diagnostics."""
        return {key: value.clone() for key, value in self._last_crop_params.items()}

    def multi_image_forward(self, obs_dict, rgb_pretransformed: bool = False):
        batch_size = None
        features = []

        obs_dict = dict(obs_dict)
        if self.temporal_consistent_crop and not rgb_pretransformed:
            obs_dict, _, _ = self.prepare_temporally_consistent_crops(obs_dict)
            rgb_pretransformed = True

        if self.use_seq:
            # input: (bs, horizon, c, h, w)
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].flatten(end_dim=1)

        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                if not rgb_pretransformed:
                    assert img.shape[1:] == self.key_shape_map[key]
                    img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map["rgb"](imgs)
            # (N,B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature, 0, 1)
            # (B,N*D)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                if not rgb_pretransformed:
                    assert img.shape[1:] == self.key_shape_map[key]
                    img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        features = torch.cat(features, dim=-1)
        return features

    def encode_rgb_features(self, obs_dict, rgb_pretransformed: bool = False):
        """Encode only RGB observations with the vision backbone.

        This is used for RGB-only future targets. Low-dimensional observations
        are intentionally ignored here, while the normal forward path still
        fuses RGB and low-dimensional inputs for policy conditioning.
        """
        if self.temporal_consistent_crop and not rgb_pretransformed:
            # This path is used when there is no paired current observation.
            # Joint future training uses encode_obs_and_future_rgb below so the
            # future target never reaches this independent-crop fallback.
            temporal_obs = {}
            for key in self.rgb_keys:
                value = obs_dict[key]
                temporal_obs[key] = value if value.dim() == 5 else value.unsqueeze(1)
            temporal_obs, _, _ = self.prepare_temporally_consistent_crops(temporal_obs)
            obs_dict = {
                key: (
                    temporal_obs[key]
                    if obs_dict[key].dim() == 5
                    else temporal_obs[key][:, 0]
                )
                for key in self.rgb_keys
            }
            rgb_pretransformed = True

        features = []

        for key in self.rgb_keys:
            img = obs_dict[key]
            if img.dim() == 5:
                b, t, c, h, w = img.shape
                img = img.reshape(b * t, c, h, w)
                if not rgb_pretransformed:
                    img = self.key_transform_map[key](img)
                model_key = "rgb" if self.share_rgb_model else key
                feature = self.key_model_map[model_key](img)
                feature = feature.reshape(b, t, -1)
            elif img.dim() == 4:
                if not rgb_pretransformed:
                    img = self.key_transform_map[key](img)
                model_key = "rgb" if self.share_rgb_model else key
                feature = self.key_model_map[model_key](img)
            else:
                raise RuntimeError(
                    f"Unexpected RGB observation shape for key {key}: {tuple(img.shape)}"
                )
            features.append(feature)

        if len(features) == 0:
            raise RuntimeError("MultiImageObsEncoder requires at least one rgb key")

        return torch.cat(features, dim=-1)

    def _forward_impl(self, obs_dict, mask=None, rgb_pretransformed: bool = False):
        ori_batch_size, ori_seq_len = self.get_batch_size(obs_dict)
        features = self.multi_image_forward(
            obs_dict, rgb_pretransformed=rgb_pretransformed
        )
        # linear embedding
        result = self.mlp(features)
        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.reshape(ori_batch_size, ori_seq_len, -1)
            else:
                result = result.reshape(ori_batch_size, -1)
        return result

    def forward(self, obs_dict, mask=None):
        return self._forward_impl(obs_dict, mask=mask, rgb_pretransformed=False)

    def encode_obs_and_future_rgb(
        self,
        obs_dict: dict[str, torch.Tensor],
        future_obs_dict: dict[str, torch.Tensor],
        mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Encode paired current/future observations after one shared crop."""
        if not self.temporal_consistent_crop:
            raise RuntimeError(
                "encode_obs_and_future_rgb requires temporal_consistent_crop=true"
            )
        prepared_obs, prepared_future, crop_params = (
            self.prepare_temporally_consistent_crops(obs_dict, future_obs_dict)
        )
        obs_embedding = self._forward_impl(
            prepared_obs, mask=mask, rgb_pretransformed=True
        )
        future_embedding = self.encode_rgb_features(
            prepared_future, rgb_pretransformed=True
        )
        return obs_embedding, future_embedding, crop_params

    @torch.no_grad()
    def output_shape(self):
        was_training = self.training
        self.eval()
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            prefix = (batch_size, 1) if self.use_seq else (batch_size,)
            this_obs = torch.zeros(prefix + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        try:
            example_output = self.multi_image_forward(example_obs_dict)
            output_shape = example_output.shape[1:]
            return output_shape[0]
        finally:
            self.train(was_training)

    @torch.no_grad()
    def rgb_feature_dim(self):
        was_training = self.training
        self.eval()
        example_obs_dict = {}
        batch_size = 1
        for key in self.rgb_keys:
            shape = self.key_shape_map[key]
            this_obs = torch.zeros(
                (batch_size,) + shape,
                dtype=self.dtype,
                device=self.device,
            )
            example_obs_dict[key] = this_obs
        try:
            example_output = self.encode_rgb_features(example_obs_dict)
            return example_output.shape[-1]
        finally:
            self.train(was_training)

    def get_batch_size(self, obs_dict):
        any_key = next(iter(obs_dict))
        any_tensor = obs_dict[any_key]
        return any_tensor.size(0), any_tensor.size(1)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
