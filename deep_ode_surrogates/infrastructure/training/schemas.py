# infrastructure/training/training_pipeline.py

from pydantic import BaseModel


class CallbackConfig(BaseModel):
    use_tensorboard: bool = True
    use_checkpoint: bool = True
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
