from os.path import join

from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from unet.data_loader.image_loader import ImageLoader
from unet.data_loader.image_loader.image_loader import calculate_image


def get_path(root: str, name: str) -> (str, str):
    base_path = join(root, name)
    return join(base_path, 'images'), join(base_path, 'masks')


class ISICImageLoader(ImageLoader):
    def __init__(self, root, dataset_config, mode='train', augmentation_prob=0.4):
        assert mode == 'train' or mode == 'valid', "Mode must be one of ['train', 'valid']"
        origin_path = None
        gt_path = None
        if mode == 'train':
            origin_path, gt_path = get_path(root, 'train')
        if mode == 'valid':
            origin_path, gt_path = get_path(root, 'val')
        super().__init__(origin_path, gt_path, dataset_config, mode, augmentation_prob)

    def get_gt_file_name(self, origin_image_name: str, extension: str) -> str:
        try:
            return self.dataset_config.gt_format.format(origin_image_name)
        except:
            return origin_image_name + "_segmentation.png"


class ISICDataLoader(DataLoader):
    @staticmethod
    def generate_loaders(config) -> [(DataLoader, DataLoader)]:
        if config.processed_image.size is None:
            _, size = calculate_image(config.image_size, config.origin_image.size)
            config.processed_image.size = size

        train_loader = ISICDataLoader(
            root=config.root_path,
            dataset_config=config,
            mode='train',
            augmentation_prob=config.augmentation_prob
        )
        valid_loader = ISICDataLoader(
            root=config.root_path,
            dataset_config=config,
            mode='valid',
            augmentation_prob=0.
        )
        return [(train_loader, valid_loader)]

    def __init__(self, root, dataset_config, mode='train',
                 augmentation_prob=0.4):
        super().__init__(
            dataset=ISICImageLoader(
                root=root,
                dataset_config=dataset_config,
                mode=mode,
                augmentation_prob=augmentation_prob
            ),
            batch_size=dataset_config.batch_size,
            shuffle=True,
            num_workers=dataset_config.num_workers
        )
