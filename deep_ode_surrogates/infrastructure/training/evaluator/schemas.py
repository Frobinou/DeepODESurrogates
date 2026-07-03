try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


from typing import Any

from pydantic import BaseModel, Field


class MetricName(StrEnum):
    MSE = "mse"


class FigureName(StrEnum):
    TRAJECTORY = "trajectory"
    PHASE_SPACE = "phase space"


class EvaluatorResults(BaseModel):
    metrics: dict[MetricName, float] = Field(default_factory=dict)
    figures: dict[FigureName, Any] = Field(default_factory=dict)
