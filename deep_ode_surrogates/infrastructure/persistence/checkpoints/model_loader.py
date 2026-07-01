# infrastructure/persistence/checkpoints/model_loader.py

import torch


def load_model_from_checkpoint(
    model,
    checkpoint_path,
    device: str = "cpu",
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model
