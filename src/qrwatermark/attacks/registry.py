from __future__ import annotations

from typing import Callable

from .compression import jpeg, jpeg2000
from .filtering import average_filter, gaussian_blur, lowpass_filter, median_filter, motion_blur, sharpen
from .geometric import crop_resize, rotation_resample, rotation_unregistered, scaling_resample, translation
from .noise import gaussian_noise, salt_pepper_noise, speckle_noise
from .occlusion import random_occlusion
from .photometric import brightness_contrast, clahe, gamma_correction, histogram_equalization

ATTACKS: dict[str, Callable] = {
    "gaussian_blur": gaussian_blur,
    "sharpen": sharpen,
    "gaussian_noise": gaussian_noise,
    "salt_pepper": salt_pepper_noise,
    "speckle_noise": speckle_noise,
    "jpeg": jpeg,
    "jpeg2000": jpeg2000,
    "lowpass": lowpass_filter,
    "median": median_filter,
    "average": average_filter,
    "motion_blur": motion_blur,
    "scale_resample": scaling_resample,
    "rotation_resample": rotation_resample,
    "rotation_unregistered": rotation_unregistered,
    "translation": translation,
    "crop_resize": crop_resize,
    "gamma": gamma_correction,
    "brightness_contrast": brightness_contrast,
    "histogram_equalization": histogram_equalization,
    "clahe": clahe,
    "occlusion": random_occlusion,
}


def apply_attack(name: str, image, **params):
    if name == "clean":
        return image.copy()
    try:
        fn = ATTACKS[name]
    except KeyError as e:
        raise KeyError(f"Unknown attack '{name}'. Available: {sorted(ATTACKS)}") from e
    return fn(image, **params)
