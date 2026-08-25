import diffusion_policy.model.vision.crop_randomizer as dmvc
import robomimic.models.base_nets as rmbn

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.temporal_consistent_crop_randomizer import (
    TemporalConsistentCropRandomizer,
)
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)


try:
    RobomimicCropRandomizer = rmbn.CropRandomizer
except AttributeError:
    # Robomimic 0.4 moved observation randomizers out of base_nets.
    from robomimic.models.obs_core import CropRandomizer as RobomimicCropRandomizer


class AlignedDiffusionTransformerHybridImagePolicy(
    DiffusionTransformerHybridImagePolicy
):
    """Transformer DP with JEPA-compatible temporal-consistent augmentation."""

    def __init__(
        self,
        *args,
        temporal_consistent_crop=True,
        eval_fixed_crop=True,
        **kwargs,
    ):
        # The stock transformer policy still references the pre-0.4
        # robomimic CropRandomizer location in its eval-fixed-crop branch.
        # Keep that branch off and perform the version-compatible replacement
        # here instead.
        super().__init__(*args, eval_fixed_crop=False, **kwargs)
        if temporal_consistent_crop:
            n_obs_steps = self.n_obs_steps
            replace_submodules(
                root_module=self.obs_encoder,
                predicate=lambda module: (
                    isinstance(
                        module,
                        (RobomimicCropRandomizer, dmvc.CropRandomizer),
                    )
                    and not isinstance(module, TemporalConsistentCropRandomizer)
                ),
                func=lambda module: TemporalConsistentCropRandomizer(
                    input_shape=module.input_shape,
                    crop_height=module.crop_height,
                    crop_width=module.crop_width,
                    num_crops=module.num_crops,
                    pos_enc=module.pos_enc,
                    temporal_group_size=n_obs_steps,
                ),
            )
        elif eval_fixed_crop:
            replace_submodules(
                root_module=self.obs_encoder,
                predicate=lambda module: isinstance(
                    module, RobomimicCropRandomizer
                ),
                func=lambda module: dmvc.CropRandomizer(
                    input_shape=module.input_shape,
                    crop_height=module.crop_height,
                    crop_width=module.crop_width,
                    num_crops=module.num_crops,
                    pos_enc=module.pos_enc,
                ),
            )
