# infrastructure/training/training_pipeline.py

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class TrainingPipeline:
    trainer: object
    dataloader: object
    callbacks: list
    evaluators: list
    experiment_path: object


class CallbackConfig(BaseModel):
    use_tensorboard: bool = True
    use_checkpoint: bool = True
    use_early_stopping: bool = True
    early_stopping_patience: int = 10