import glob
import cv2
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset


from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms.functional import center_crop


class DataGenerator(Dataset):
    def __init__(self, img_dir, data_len):
        self.img_dir = img_dir
        self.data_len = data_len

        self.transform = transforms.Compose([
            transforms.Resize((572, 572))
        ])
        self.transform_label = transforms.Compose([])

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_path = f'{self.img_dir}/{index}.jpg'
        image = cv2.imread(img_path)
        image = cv2.resize(image, (572, 572)).astype(np.float32)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        mask_path = f'{self.img_dir}/{index}_mask.jpg'
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (572, 572)).astype(np.longlong)
        mask = np.where(mask > 100, 0, 1)
        mask = torch.from_numpy(mask)
        mask = center_crop(mask, (388, 388))
        
        return image, mask
