# application/train.py

from deep_ode_surrogates.application.config.training import TrainingPipeline


class TrainUseCase:
    def execute(
        self,
        training_pipeline: TrainingPipeline,
        epochs: int,
    ) -> None:
        training_pipeline.trainer.callbacks.extend(training_pipeline.callbacks)
        training_pipeline.trainer.evaluators.extend(training_pipeline.evaluators)

        training_pipeline.trainer.fit(
            dataloader=training_pipeline.dataloader,
            epochs=epochs,
        )
