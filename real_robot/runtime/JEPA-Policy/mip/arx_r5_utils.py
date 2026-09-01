"""Helpers shared by the offline ARX R5 training path."""


def is_arx_r5_task(task_config) -> bool:
    return getattr(task_config, "env_type", None) == "arx_r5"
