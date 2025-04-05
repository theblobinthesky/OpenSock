import wandb
from typing import Any
from .config import DataConfig, TrainConfig, ModelConfig

def _config_to_dict(config) -> dict[str, Any]:
    return { config.__class__.__name__: vars(config) }

class Logger:
    def __init__(self, project: str, dconfig: DataConfig, tconfig: TrainConfig, mconfig: ModelConfig):
        wandb.init(
            project=project,
            config=_config_to_dict(dconfig) | _config_to_dict(tconfig) | _config_to_dict(mconfig)
        )

    def log(self, data: dict[str, Any]):
        wandb.log(data)


class MockLogger:
    def log(self, data: dict[str, Any]):
        pass
