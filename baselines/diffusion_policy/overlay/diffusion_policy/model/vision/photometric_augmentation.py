"""Fast batch-level photometric augmentation for robot RGB observations."""

import torch
from torchvision.transforms import functional as TVF


def augment_robot_rgb_batch(obs, config):
    """Augment all cameras and time steps with one shared transform on GPU."""
    if not bool(config.get("gpu_photometric_aug_enabled", False)):
        return obs
    probability = float(config.get("photometric_aug_probability", 0.8))
    if torch.rand((), device="cpu").item() >= probability:
        return obs

    def uniform(low, high):
        return float(torch.empty((), device="cpu").uniform_(low, high).item())

    brightness_delta = float(config.get("photometric_brightness", 0.2))
    contrast_delta = float(config.get("photometric_contrast", 0.2))
    saturation_delta = float(config.get("photometric_saturation", 0.1))
    hue_delta = float(config.get("photometric_hue", 0.03))
    brightness = uniform(1.0 - brightness_delta, 1.0 + brightness_delta)
    contrast = uniform(1.0 - contrast_delta, 1.0 + contrast_delta)
    saturation = uniform(1.0 - saturation_delta, 1.0 + saturation_delta)
    hue = uniform(-hue_delta, hue_delta)
    gamma = uniform(
        float(config.get("photometric_gamma_min", 0.8)),
        float(config.get("photometric_gamma_max", 1.2)),
    )

    result = dict(obs)
    for key, value in obs.items():
        if not key.endswith("_image"):
            continue
        image = TVF.adjust_brightness(value, brightness)
        image = TVF.adjust_contrast(image, contrast)
        image = TVF.adjust_saturation(image, saturation)
        image = TVF.adjust_hue(image, hue)
        image = TVF.adjust_gamma(image, gamma)
        result[key] = image.clamp_(0.0, 1.0)
    return result
