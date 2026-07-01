# application/infer_surrogate.py

import torch


class InferSurrogateUseCase:
    def execute(
        self,
        model: torch.nn.Module,
        t: torch.Tensor,
        device: str = "cpu",
    ) -> torch.Tensor:
        model.to(device)
        model.eval()

        t = t.to(device)

        with torch.no_grad():
            y_pred = model(t)

        return y_pred.cpu()
