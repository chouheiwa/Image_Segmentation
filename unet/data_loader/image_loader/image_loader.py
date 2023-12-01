from os import listdir
from os.path import join, isfile, splitext, basename
import random

from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class ImageLoader(Dataset):
    def __init__(
            self,
            origin_image_path,
            gt_image_path,
            image_size=224,
            mode='train',
            augmentation_prob=0.4,
            support_types: [str] = None
    ):
        """Initializes image paths and preprocessing module."""
        if support_types is None:
            support_types = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF']
        support_types = set(support_types)
        self.origin_image_path = origin_image_path
        self.ground_truth_path = gt_image_path
        self.image_paths = [join(origin_image_path, f) for f in listdir(origin_image_path)
                            if isfile(join(origin_image_path, f)) and splitext(f)[1][1:] in support_types]
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def get_gt_file_name(self, origin_image_name: str, extension: str) -> str:
        assert False, "Need to implement this method in child class"
        return ""

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        origin_image_name, extension = splitext(basename(image_path))
        filename = self.get_gt_file_name(origin_image_name, extension)
        GT_path = join(self.ground_truth_path, filename)

        image = Image.open(image_path)
        GT = Image.open(GT_path)

        aspect_ratio = image.size[1] / image.size[0]

        Transform = []

        ResizeRange = random.randint(300, 320)
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))
            CropRange = random.randint(250, 270)
            Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
            Transform = T.Compose(Transform)

            image = Transform(image)
            GT = Transform(GT)

            ShiftRange_left = random.randint(0, 20)
            ShiftRange_upper = random.randint(0, 20)
            ShiftRange_right = image.size[0] - random.randint(0, 20)
            ShiftRange_lower = image.size[1] - random.randint(0, 20)
            image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            GT = GT.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

            image = Transform(image)

            Transform = []

        Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        GT = Transform(GT)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)