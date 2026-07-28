"""Helpers for keeping CUDA compute and EGL render device ids separate."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


EGL_DEVICE_ENV = "JEPA_POLICY_EGL_DEVICE_ID"


def configured_egl_device_id() -> int | None:
    """Return the explicitly configured EGL device, if one was requested."""
    raw_device_id = os.environ.get(EGL_DEVICE_ENV)
    if raw_device_id is None:
        return None
    try:
        device_id = int(raw_device_id)
    except ValueError as exc:
        raise ValueError(f"{EGL_DEVICE_ENV} must be a non-negative integer") from exc
    if device_id < 0:
        raise ValueError(f"{EGL_DEVICE_ENV} must be a non-negative integer")
    return device_id


def install_robosuite_egl_device_override() -> int | None:
    """Make robosuite honor JEPA's EGL id instead of CUDA_VISIBLE_DEVICES.

    robosuite 1.4 and 1.5 both prioritize a single-value
    CUDA_VISIBLE_DEVICES inside their EGL display helper, even when the caller
    explicitly passes render_gpu_device_id. That is incorrect for software
    Mesa, whose only EGL device is index 0. Patch the lazy EGL display factory
    in this process while leaving CUDA visibility unchanged for PyTorch.
    """
    configured_device_id = configured_egl_device_id()
    if configured_device_id is None:
        return None

    from robosuite.renderers.context import egl_context

    installed_device_id = getattr(
        egl_context, "_jepa_policy_egl_device_override", None
    )
    if installed_device_id is not None:
        if installed_device_id != configured_device_id:
            raise RuntimeError(
                "robosuite EGL device was already configured as "
                f"{installed_device_id}, cannot change it to {configured_device_id}"
            )
        return configured_device_id

    def create_initialized_egl_device_display(device_id=0):
        del device_id
        all_devices = egl_context.EGL.eglQueryDevicesEXT()
        if not 0 <= configured_device_id < len(all_devices):
            raise RuntimeError(
                f"{EGL_DEVICE_ENV} must be between 0 and "
                f"{len(all_devices) - 1} (inclusive), got {configured_device_id}"
            )

        device = all_devices[configured_device_id]
        display = egl_context.EGL.eglGetPlatformDisplayEXT(
            egl_context.EGL.EGL_PLATFORM_DEVICE_EXT, device, None
        )
        if (
            display == egl_context.EGL.EGL_NO_DISPLAY
            or egl_context.EGL.eglGetError() != egl_context.EGL.EGL_SUCCESS
        ):
            return egl_context.EGL.EGL_NO_DISPLAY

        try:
            initialized = egl_context.EGL.eglInitialize(display, None, None)
        except egl_context.error.GLError:
            return egl_context.EGL.EGL_NO_DISPLAY
        if (
            initialized == egl_context.EGL.EGL_TRUE
            and egl_context.EGL.eglGetError() == egl_context.EGL.EGL_SUCCESS
        ):
            return display
        return egl_context.EGL.EGL_NO_DISPLAY

    egl_context.create_initialized_egl_device_display = (
        create_initialized_egl_device_display
    )
    egl_context._jepa_policy_egl_device_override = configured_device_id
    return configured_device_id


@contextmanager
def override_robomimic_egl_probe() -> Iterator[None]:
    """Make robomimic use the configured EGL id instead of a CUDA card id.

    robomimic's EnvRobosuite asks egl_probe for a render device and overwrites
    the caller's render_gpu_device_id. On software Mesa there is one EGL device
    (index 0), even when CUDA_VISIBLE_DEVICES contains a physical card such as 3.
    """
    device_id = configured_egl_device_id()
    if device_id is None:
        yield
        return

    install_robosuite_egl_device_override()

    import egl_probe

    original_get_available_devices = egl_probe.get_available_devices
    egl_probe.get_available_devices = lambda: [device_id]
    try:
        yield
    finally:
        egl_probe.get_available_devices = original_get_available_devices
