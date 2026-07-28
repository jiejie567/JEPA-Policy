"""Small helpers shared by the isolated RoboTwin training path."""


def is_robotwin_task(task_config) -> bool:
    return getattr(task_config, "env_type", None) == "robotwin"
