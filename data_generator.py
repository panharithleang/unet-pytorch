from numpy import random
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset


from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms.transforms import RandomVerticalFlip


class DataGenerator(Dataset):
    def __init__(self, img_dir, data_len, is_train_set):
        self.img_dir = img_dir
        self.data_len = data_len
        self.is_train_set = is_train_set

        self.train_set_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(572)
            transforms.Resize([572, 572])
        ])
        self.train_set_target_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize([388, 388]),
            # transforms.CenterCrop([388, 388])
            # transforms.RandomResizedCrop(572)
        ])
        self.validation_set_transform = transforms.Compose([
            transforms.Resize([572, 572])
        ])
        self.validation_set_target_transform = transforms.Compose([
            transforms.Resize([388, 388]),
            # transforms.CenterCrop([388, 388])
        ])

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        # make a seed with numpy generator
        seed = random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)

        img_path = f'{self.img_dir}/{index}.jpg'

        img = read_image(img_path)
        if self.is_train_set:
            img = self.train_set_transform(img)
        else:
            img = self.validation_set_transform(img)
        img = img.float()

        random.seed(seed)
        torch.manual_seed(seed)

        mask_path = f'{self.img_dir}/{index}_mask.jpg'
        mask = read_image(mask_path)
        if self.is_train_set:
            mask = self.train_set_target_transform(mask)
        else:
            mask = self.validation_set_target_transform(mask)
        mask = torch.where(mask > 100, 1, 0)
        mask = mask[0]

        return img, mask
