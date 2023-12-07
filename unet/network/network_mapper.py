from typing import List, Type
from functools import reduce

from .network_type import NetworkType
from .u_net import UNet
from .att_u_net import AttUNet
from .r2u_net import R2UNet
from .r2_att_u_net import R2AttUNet
from .trans_u_net import TransUNet

support_list: List[Type[NetworkType]] = [
    UNet,
    AttUNet,
    R2UNet,
    R2AttUNet,
    TransUNet,
]

network_mapper = reduce(
    lambda dic, item: {**dic, item.name(): item.create_model},
    support_list,
    {}
)

data_mapper = reduce(
    lambda dic, item: {**dic, item.name(): item.cache_model_name},
    support_list,
    {}
)


def get_support_list():
    return list(network_mapper.keys())


def get_network(config, dataset_config, device, load_pretrained_model=True):
    return network_mapper[config["model_type"]](config, dataset_config, device,
                                                load_pretrained_model=load_pretrained_model)


def get_cached_pretrained_model(config):
    return data_mapper[config.network.model_type](config)
