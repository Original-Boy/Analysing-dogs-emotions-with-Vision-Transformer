from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

class MyDataSet(Dataset):
    """Customized data sets"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB is the color picture, L is the grayscale picture
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
            
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # The official implementation of default_collate can be found in
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels