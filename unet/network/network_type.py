from torch import nn


class NetworkType(nn.Module):
    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def create_model(cls, config):
        raise NotImplementedError
