# ODEs
# import deep_ode_surrogates.domain.models.model_diff
# Losses
import deep_ode_surrogates.domain.losses.pinn_losses  # noqa: F401

# Models
import deep_ode_surrogates.domain.models.model_PINN  # noqa: F401
import deep_ode_surrogates.domain.odes.ode_lotka_voltera  # noqa: F401

# DataLoaders
import deep_ode_surrogates.infrastructure.data.parquet.parquet_dataloader  # noqa: F401


def bootstrap():
    pass
