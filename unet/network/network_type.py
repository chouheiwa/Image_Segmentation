from torch import nn


class NetworkType(nn.Module):
    base_config = None

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def create_model(cls, config, dataset_config, device, **kwargs):
        model = cls.initialize_model(config, dataset_config, device, **kwargs)
        model.base_config = config
        model.to(device)
        return model

    @classmethod
    def initialize_model(cls, config, dataset_config, device, **kwargs):
        raise NotImplementedError

    @classmethod
    def cache_model_name(cls, base_config) -> str:
        return base_config.network.model_type
