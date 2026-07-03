from dataclasses import dataclass, field

import torch

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class LossName(StrEnum):
    TOTAL = "total"
    PHYSICS = "physics"
    IC = "ic"
    DATA = "data"
    RESIDUALS = "residuals"


@dataclass
class EpochStats:
    loss_sums: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in LossName})
    loss_counts: dict[str, int] = field(default_factory=lambda: {name: 0.0 for name in LossName})
    num_batches: int = 0
    residuals_sum: torch.Tensor | None = None
    residuals_count: int = 0

    def update(self, loss_dict: dict):
        self.num_batches += 1

        for key in self.loss_sums:
            value = loss_dict.get(key)
            if value is not None:
                self.loss_sums[key] += value.detach().item()
                self.loss_counts[key] += 1

        residuals = loss_dict.get("residuals")
        if residuals is not None:
            residuals_mean = residuals.detach().mean(dim=0).cpu()

            self.residuals_sum = (
                residuals_mean
                if self.residuals_sum is None
                else self.residuals_sum + residuals_mean
            )
            self.residuals_count += 1

    def as_loss_state(self) -> dict:
        state = {
            key: (
                None
                if self.loss_counts[key] == 0
                else torch.tensor(self.loss_sums[key] / self.loss_counts[key])
            )
            for key in self.loss_sums
        }

        state["residuals"] = (
            None if self.residuals_count == 0 else self.residuals_sum / self.residuals_count
        )

        return state

    @property
    def epoch_loss(self) -> float:
        if self.loss_counts["total"] == 0:
            return 0.0

        return self.loss_sums["total"] / self.loss_counts["total"]
