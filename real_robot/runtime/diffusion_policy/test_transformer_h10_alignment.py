import torch

from diffusion_policy.model.vision.temporal_consistent_crop_randomizer import (
    TemporalConsistentCropRandomizer,
)
from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)


def test_transformer_accepts_horizon_10():
    model = TransformerForDiffusion(
        input_dim=7,
        output_dim=7,
        horizon=10,
        n_obs_steps=2,
        cond_dim=32,
        n_layer=2,
        n_head=2,
        n_emb=32,
        causal_attn=False,
        obs_as_cond=True,
    )
    output = model(torch.randn(3, 10, 7), 5, torch.randn(3, 2, 32))
    assert output.shape == (3, 10, 7)


def test_temporal_crop_reuses_offsets_within_sample():
    torch.manual_seed(7)
    base = torch.arange(84 * 84, dtype=torch.float32).reshape(1, 1, 84, 84)
    images = torch.cat((base, base, base + 100000, base + 100000), dim=0)
    randomizer = TemporalConsistentCropRandomizer(
        input_shape=(1, 84, 84),
        crop_height=76,
        crop_width=76,
        num_crops=1,
        temporal_group_size=2,
    )
    randomizer.train()
    crops = randomizer.forward_in(images)
    assert torch.equal(crops[0], crops[1])
    assert torch.equal(crops[2] - 100000, crops[3] - 100000)
