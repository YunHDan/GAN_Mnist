import numpy as np
import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms as transforms
import os


class GanDataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir

        self.TestDataset = []
        for i in range(10):
            data_folder = os.path.join(self.img_dir, str(i))
            for filename in os.listdir(data_folder):
                if os.path.splitext(filename)[1] == '.png':
                    img_path = os.path.join(data_folder, filename)
                    image = Image.open(img_path)
                    self.TestDataset.append((image, img_path))
        self.count = len(self.TestDataset)
        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transforms_list)

    def load_img_transform(self, file):
        img_norm = self.transform(file).numpy()
        return img_norm

    def __getitem__(self, item):
        trans_img = self.load_img_transform(self.TestDataset[item][0])
        trans_img = np.asarray(trans_img, dtype=np.float32)
        return torch.from_numpy(trans_img), self.TestDataset[item][1]

    def __len__(self):
        return self.count
