# data/transforms.py

import torchvision.transforms as T
import cv2
import numpy as np


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
def resize_and_normalize(
    img,
    target_size=(640, 640)
):
    """
    Resize image and normalize to [0,1]
    Output: C x H x W
    """
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return img