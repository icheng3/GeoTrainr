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

COORD_REF = np.array((50, 10))

class GeoDataset(data.Dataset):
    def __init__(self, paths, transform, trim=False):
        self.cls = []
        self.dataset = []
        for image in paths:
            name = os.path.basename(image)[:-4]
            country, coord = name.split("_")
            lat, lng = coord.split(",")

            latlng = np.array([float(lat), float(lng)])
            if trim and latlng[1]>50:
                continue
            self.dataset.append( (country, latlng-COORD_REF, image) )
            if country not in self.cls:
                self.cls.append(country)

        self.cls_map = {c: i for i, c in enumerate(self.cls)}
        print(len(self.dataset))
        self.trans = transform
        self.n_sample = len(self.dataset)


    def __getitem__(self, index):
        country, coord, img_path = self.dataset[index]
        img = Image.open(img_path)
        return self.trans(img), torch.Tensor(coord), self.cls_map[country]

    def __len__(self):
        return self.n_sample


def build_geo_dataset(args, trim=False):
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

    trainset = GeoDataset(images[:split], train_transform, trim)
    testset = GeoDataset(images[split:], test_transform, trim)
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
            scale = (0.4, 1.0)
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

def create_anchor_transform(args):
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    t = [
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    return transforms.Compose(t)


anchor_samples = [
    "AD_42.528,1.56927.png",
    "AL_41.32654,19.82209.png",
    "AT_47.73333,14.21667.png",
    "BA_43.91194,18.08083.png",
    "BE_50.78263,4.5334.png",
    "BG_42.71231,25.3329.png",
    "BY_53.0245,26.3403.png",
    "CH_46.90981,8.11206.png",
    "CY_35.119479999999996,33.28853.png",
    "CZ_49.73456,15.29297.png",
    "DE_50.39996,9.98198.png",
    "DK_55.80849,10.581669999999999.png",
    "EE_58.63053,25.55402.png",
    "ES_39.68888,-3.50281.png",
    "FI_61.929730000000006,25.15144.png",
    "FR_46.91745,2.49814.png",
    "GB_52.81773,-1.76009.png",
    "GR_37.97451,23.51769.png",
    "HR_44.655,15.95083.png",
    "HU_47.25,19.06667.png",
    "IE_53.32528000000001,-7.979439999999999.png",
    "IS_64.13267,-20.30651.png",
    "IT_43.43218,11.77323.png",
    "LI_47.17556,9.57287.png",
    "LT_55.41019,23.7299.png",
    "LU_49.64506,6.12932.png",
    "LV_57.0619,24.84465.png",
    "MC_43.74041,7.42311.png",
    "MD_47.01095,28.85176.png",
    "ME_42.39333,18.89028.png",
    "MK_41.63468,21.40268.png",
    "MT_35.94556,14.38972.png",
    "NL_52.1738,5.48497.png",
    "NO_62.20631,10.63725.png",
    "PL_51.85225,19.59197.png",
    "PT_39.66978,-8.9958.png",
    "RO_45.68811,24.97548.png",
    "RS_44.24947,20.39613.png",
    "RU_54.1766,37.8881.png",
    "SE_59.06565,15.337470000000001.png",
    "SI_46.05804,14.82515.png",
    "SK_48.56315,19.3029.png",
    "SM_43.90867,12.44808.png",
    "UA_48.57325,29.71874.png",
    "VA_41.90394,12.45401.png",
    "XK_42.54018,20.28793.png",
]