# -*- coding: utf-8 -*-
"""
created by @Esha Singh 
November 18, 2020

"""

import os
import matplotlib.pyplot as plt
import cv2
import random
import albumentations as A
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm as tqdm
from torch.utils.data import Dataset

class LandmarkDataset(Dataset):
    def __init__(self, csv, mode, transform=None):

        self.csv = csv.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)[:,:,::-1]

        if self.transform is not None:
            if self.mode == 'test':
                res_0 = self.transform[0](image=image)
                image_0 = res_0['image'].astype(np.float32)
                image_0 = image_0.transpose(2, 0, 1)
                res_1 = self.transform[1](image=image)
                image_1 = res_1['image'].astype(np.float32)
                image_1 = image_1.transpose(2, 0, 1)
                res_2 = self.transform[2](image=image)
                image_2 = res_2['image'].astype(np.float32)
                image_2 = image_2.transpose(2, 0, 1)
            else:
                res = self.transform(image=image)
                image = res['image'].astype(np.float32)
                image = image.transpose(2, 0, 1)
        else:
            image = image.astype(np.float32)
            image = image.transpose(2, 0, 1)

        if self.mode == 'test':
            return torch.tensor(image_0), torch.tensor(image_1), torch.tensor(image_2)
        else:
            return torch.tensor(image), torch.tensor(row.landmark_id)


def get_transforms(image_size, mode='train'):
    if mode == 'train':
        """
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
        """
        
        transform_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
            A.Resize(image_size, image_size),
            A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
            A.Normalize()
        ])
        
        transform_val = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize()
        ])
    
        return transform_train, transform_val
    
    if mode =='test':
        transform_test_0 = A.Compose([
            A.Resize(int(image_size/2**0.5), int(image_size/2**0.5)),
            A.Normalize()
        ])
        
        transform_test_1 = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize()
        ])
        
        transform_test_2 = A.Compose([
            A.Resize(int(image_size*2**0.5), int(image_size*2**0.5)),
            A.Normalize()
        ])
        
        return transform_test_0, transform_test_1, transform_test_2
    
    
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)


def get_df(data_dir, csv_fname, mode='train'):
    df = pd.read_csv(csv_fname)
    # add file path to df
    df['filepath'] = df['id'].apply(lambda x: os.path.join(data_dir, x[0], x[1], x[2], f'{x}.jpg'))
    
    if mode == 'train':
        landmark_id2idx = {landmark_id: idx for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
        df['landmark_id'] = df['landmark_id'].map(landmark_id2idx)
        num_classes = df.landmark_id.nunique()
    
        return df, num_classes
    
    if mode == 'test':
        return df

    

####### Testing/Visualisation code ########
# random.seed(42) 
# image = cv2.imread('test_img.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# augmented_image = transforms_train(image=image)['image']
# visualize(augmented_image)