import torch


class BoundaryLoss(torch.nn.Module):
    """Some Information about BoundaryLoss"""

    def __init__(self, parameters=None):
        super().__init__()

        if parameters is None:
            parameters = {}
        self.gamma = parameters.get("gamma", 1.5)
        root = parameters.get("root", "l2")
        if root == "l2":
            self.calc_root_loss = lambda p, t: torch.abs(p - t) ** 2
        elif root == "l1":
            self.calc_root_loss = lambda p, t: torch.abs(p - t)
