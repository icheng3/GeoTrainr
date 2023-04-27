import numpy as np
import torch
import torch.utils.data as data
import glob
import torchvision.transforms as transforms
from PIL import Image
import random
import os


class GeoDataset(data.Dataset):
    def __init__(self, path, transform):
        images = glob.glob(path+'/*.png')
        self.cls = []
        self.dataset = []
        for image in images:
            name = os.path.basename(image)[:-4]
            country, coord = name.split("_")
            lat, lng = coord.split(",")
            self.dataset.append( (country, np.array([float(lat), float(lng)]), image) )
            if country not in self.cls:
                self.cls.append(country)

        self.cls_map = {}
        for i,c in enumerate(self.cls):
            self.cls_map[c] = i

        print(len(self.dataset))
        self.trans = transform


    def __getitem__(self, index):
        country, coord, img_path = self.dataset[index]
        img = Image.open(img_path)
        return self.trans(img), torch.Tensor(coord), self.cls_map[country]

    def __len__(self):
        return len(self.dataset)

    