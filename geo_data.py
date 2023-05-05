import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import glob
import torchvision.transforms as transforms
from PIL import Image
import os
import random
from timm.data import create_transform
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


class GeoDataset(data.Dataset):
    def __init__(self, paths, transform):
        self.cls = []
        self.dataset = []
        for image in paths:
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


def build_geo_dataset(args):
    train_transform = build_transform(True, args)
    test_transform = build_transform(False, args)

    print("Train Transform = ")
    if isinstance(train_transform, tuple):
        for trans in train_transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in train_transform.transforms:
            print(t)
    print("---------------------------")
    print("Test Transform = ")
    if isinstance(test_transform, tuple):
        for trans in test_transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in test_transform.transforms:
            print(t)
    print("---------------------------")

    root = args.data_path
    images = glob.glob(root+'/*.png')
    total = len(images)
    split = int(0.8*total)
    random.shuffle(images)

    trainset = GeoDataset(images[:split], train_transform)
    testset = GeoDataset(images[split:], test_transform)
    nb_classes = len(trainset.cls)
    return trainset, testset, nb_classes
    


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
            scale = (0.8, 1.0)
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
