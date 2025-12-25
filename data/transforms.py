# data/transforms.py

import torchvision.transforms as T


def build_transforms(img_size, train=True):
    if train:
        return T.Compose([
            T.Resize(img_size),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
        ])
