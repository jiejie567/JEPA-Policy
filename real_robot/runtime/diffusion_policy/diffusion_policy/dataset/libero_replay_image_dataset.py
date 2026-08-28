from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    RobomimicReplayImageDataset,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


class LiberoReplayImageDataset(RobomimicReplayImageDataset):
    """LIBERO HDF5 dataset using Diffusion Policy's replay-buffer cache."""

    def __init__(self, *args, **kwargs):
        kwargs["abs_action"] = False
        super().__init__(*args, **kwargs)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = get_identity_normalizer_from_stat(
            array_to_stats(self.replay_buffer["action"])
        )

        for key in self.lowdim_keys:
            normalizer[key] = get_range_normalizer_from_stat(
                array_to_stats(self.replay_buffer[key])
            )

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer
