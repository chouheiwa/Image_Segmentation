from os.path import join

from unet.data_loader.image_loader import ImageLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader


def get_path(root: str) -> (str, str):
    return join(root, 'image'), join(root, 'mask')


class DDTIImageLoader(ImageLoader):
    def __init__(self, root, dataset_config):
        super().__init__(join(root, 'image'), join(root, 'mask'), dataset_config, 'Train')

    def get_gt_file_name(self, origin_image_name: str, extension: str) -> str:
        return origin_image_name + extension


class DDTIDataLoader(DataLoader):
    @staticmethod
    def generate_loaders(config):
        loader = DDTIImageLoader(
            root=config.root_path,
            dataset_config=config
        )
        seed = config.seed if 'seed' in config else 42
        fold = KFold(n_splits=5, shuffle=True, random_state=seed)

        data_list = []

        for train_idx, valid_idx in fold.split(loader):
            train_set = DDTIDataLoader(
                dataset=Subset(loader, train_idx),
                dataset_config=config,
                shuffle=True
            )
            valid_set = DDTIDataLoader(
                dataset=Subset(loader, valid_idx),
                dataset_config=config,
                shuffle=False
            )
            item = train_set, valid_set
            data_list.append(item)
        return data_list

    def __init__(self, dataset, dataset_config, shuffle=True):
        super().__init__(
            dataset=dataset,
            batch_size=dataset_config.batch_size,
            shuffle=shuffle,
            num_workers=dataset_config.num_workers
        )
