from typing import Tuple

from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from unet.data_loader.image_loader.isic.ddti.ddti_data_loader import DDTIDataLoader
from unet.data_loader.image_loader.isic.isic_2018 import ISIC2018DataLoader
from unet.data_loader.image_loader.test_dataset.isic_data_loader import ISICDataLoader

dataloader_mapping = {
    'isic-2018-task-1': ISIC2018DataLoader.generate_loaders,
    'isic-2018-task': ISICDataLoader.generate_loaders,
    'isic-2017-task': ISICDataLoader.generate_loaders,
    'isic-2016-task': ISICDataLoader.generate_loaders,
    'ddti-task': DDTIDataLoader.generate_loaders,
}


def get_data_loader(config: ConfigDict) -> [Tuple[DataLoader, DataLoader]]:
    return dataloader_mapping[config.type](config)
