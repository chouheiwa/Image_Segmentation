from torch import nn, Tensor


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """

    def forward(self, inputT: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(inputT.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(inputT, target.long())
