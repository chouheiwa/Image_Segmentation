from typing import Tuple

from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from unet.data_loader.image_loader.isic.isic_2018 import ISIC2018DataLoader

dataloader_mapping = {
    'isic-2018-task-1': ISIC2018DataLoader.generate_loaders
}


def get_data_loader(config: ConfigDict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return dataloader_mapping[config.type](config)
