import torch


def load_model(model, model_path, strict=False):
    """
    Load model from saved weights.
    """
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=strict)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=strict)
    return model
