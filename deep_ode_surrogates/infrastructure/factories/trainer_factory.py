from typing import Literal

import torch

from deep_ode_surrogates.application.config.experiment import PhysicsWeights
from deep_ode_surrogates.application.config.ode import ODESConfig
from deep_ode_surrogates.application.config.training import TrainingConfig
from deep_ode_surrogates.infrastructure.registries.loss_registry import loss_registry
from deep_ode_surrogates.infrastructure.registries.model_registry import model_registry
from deep_ode_surrogates.infrastructure.registries.ode_registry import ode_registry
from deep_ode_surrogates.infrastructure.training.torch.trainer import Trainer


def build_trainer(
    ode_config: ODESConfig,
    loss_config: PhysicsWeights,
    training_config: TrainingConfig,
    device: Literal["cpu", "cuda"] = "cpu",
) -> Trainer:
    """Build a fully configured :class:`~src.core.trainer.Trainer` from config objects.

        Resolves each component (ODE, model, loss, optimizer) from the global
        :data:`~src.core.registry.REGISTRY` and assembles them into a ready-to-use
        trainer. No manual instantiation is required in the calling code.

        Args:
            ode_config: ODE configuration. Must expose:

                - ``ode_name`` *(str)*: registry key for the ODE class.
                - ``parameters`` *(dict)*: passed as ``params=`` to the ODE constructor.
                - ``dimension`` *(int)*: dimension of the ODE system, used to build the model.


            loss_config: Loss configuration. Must expose:

                - ``name`` *(str)*: registry key for the loss class.
                - ``lambda_ode`` *(float)*: weight for the physics residual term.
                - ``lambda_data`` *(float)*: weight for the supervised data term.

            training_config: Training hyperparameters. Must expose:

                - ``l_r`` *(float)*: learning rate passed to ``torch.optim.Adam``.
                - model_name *(str)*: registry key for the model class.

            device: Target device for the trainer (``"cpu"`` or ``"cuda"``).
                Defaults to ``"cpu"``.

        Returns:
            A :class:`~src.core.trainer.Trainer` instance with model, optimizer,
            and loss function attached, ready to call ``.fit()``.

        Example:
    ```python
            trainer = build_trainer(
                ode_config=ode_config,
                loss_config=loss_config,
                training_config=training_config,
                device="cuda",
            )
            trainer.fit(dataloader=make_dataloader(data_config), epochs=2000)
    ```
    """
    ode = ode_registry.create(ode_config.ode_name, params=ode_config.parameters)

    model = model_registry.create(
        training_config.model_name,
        input_dim=1,  # Because for ODEs, the input is usually time t, which is 1D
        output_dim=ode_config.dimension,
    )

    loss = loss_registry.create(name=loss_config.name, ode=ode, loss_config=loss_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        t=torch.linspace(ode_config.t_span[0], ode_config.t_span[1], ode_config.grid_size),
    )
