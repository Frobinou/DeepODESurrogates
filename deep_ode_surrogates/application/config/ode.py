from pydantic import BaseModel, Field, model_validator

from deep_ode_surrogates.domain.odes import AvailablesODE


class ODESConfig(BaseModel):
    """Physics side: ODE definition + simulation grid."""

    ode_name: AvailablesODE | None = None
    parameters: BaseModel
    t_span: tuple[float, float] = (0.0, 10.0)
    grid_size: int = Field(1000, gt=0)
    initial_conditions: list[float] = Field(default_factory=list)
    dimension: int = Field(1, gt=0)  # for vector-valued ODEs

    @model_validator(mode="after")
    def check_initial_conditions(self) -> "ODESConfig":
        if len(self.initial_conditions) != self.dimension:
            raise ValueError(
                f"initial_conditions has {len(self.initial_conditions)} elements "
                f"but dimension={self.dimension}."
            )
        return self
