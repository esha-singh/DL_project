# -*- coding: utf-8 -*-
"""
created by @Esha Singh 
November 18, 2020

"""


import matplotlib.pyplot as plt
import cv2
import random
import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LandmarkDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None):

        self.csv = csv.reset_index()
        self.split = split
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)[:,:,::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        if self.mode == 'test':
            return torch.tensor(image)
        else:
            return torch.tensor(image), torch.tensor(row.landmark_id)

def get_transforms(image_size):
    transform_train = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.ImageCompression(quality_lower=99, quality_upper=100),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
        A.Normalize()
    ])

    transform_val = albumentations.Compose([
              albumentations.Resize(image_size, image_size),
              albumentations.Normalize()
          ])

    return transform_train, transform_val

 def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)


####### Testing/Visualisation code ########
# random.seed(42) 
# image = cv2.imread('test_img.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# augmented_image = transforms_train(image=image)['image']
# visualize(augmented_image)