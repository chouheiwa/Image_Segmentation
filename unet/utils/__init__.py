import torch
from .path_helper import get_root_dir, get_pre_train_model
from .all_gather_grad import AllGatherGrad


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
