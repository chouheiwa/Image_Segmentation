from torch import nn


class NetworkType(nn.Module):
    base_config = None

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def create_model(cls, config, dataset_config, device, **kwargs):
        raise NotImplementedError

    @classmethod
    def cache_model_name(cls, base_config) -> str:
        return base_config.network.model_type
