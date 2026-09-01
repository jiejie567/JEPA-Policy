"""Small helpers shared by the isolated RoboCasa training path."""


def is_robocasa_task(task_config) -> bool:
    return getattr(task_config, "env_type", None) == "robocasa"
