"""Fast batch-level photometric augmentation for robot RGB observations."""

import torch
from torchvision.transforms import functional as TVF


def augment_robot_rgb_batch(obs, future_obs, config):
    """Augment all cameras and time steps with one shared transform on GPU."""
    if not bool(getattr(config, "gpu_photometric_aug_enabled", False)):
        return obs, future_obs
    probability = float(getattr(config, "photometric_aug_probability", 0.8))
    if torch.rand((), device="cpu").item() >= probability:
        return obs, future_obs

    def uniform(low, high):
        return float(torch.empty((), device="cpu").uniform_(low, high).item())

    brightness_delta = float(getattr(config, "photometric_brightness", 0.2))
    contrast_delta = float(getattr(config, "photometric_contrast", 0.2))
    saturation_delta = float(getattr(config, "photometric_saturation", 0.1))
    hue_delta = float(getattr(config, "photometric_hue", 0.03))
    brightness = uniform(1.0 - brightness_delta, 1.0 + brightness_delta)
    contrast = uniform(1.0 - contrast_delta, 1.0 + contrast_delta)
    saturation = uniform(1.0 - saturation_delta, 1.0 + saturation_delta)
    hue = uniform(-hue_delta, hue_delta)
    gamma = uniform(
        float(getattr(config, "photometric_gamma_min", 0.8)),
        float(getattr(config, "photometric_gamma_max", 1.2)),
    )

    def apply(mapping):
        if mapping is None:
            return None
        result = dict(mapping)
        for key, value in mapping.items():
            if not key.endswith("_image"):
                continue
            image = value.add(1.0).mul(0.5)
            image = TVF.adjust_brightness(image, brightness)
            image = TVF.adjust_contrast(image, contrast)
            image = TVF.adjust_saturation(image, saturation)
            image = TVF.adjust_hue(image, hue)
            image = TVF.adjust_gamma(image, gamma)
            result[key] = image.clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)
        return result

    return apply(obs), apply(future_obs)
