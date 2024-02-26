import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from data_preparation.config import *
from data_preparation.action_dataset import ActionDataset


def get_transformer(phase):
    valid_trans = A.Compose([
        A.Resize(height=CFG.height, width=CFG.width, interpolation=cv2.INTER_LINEAR), 
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(p=1.0)
    ])

    if phase == 'train':
        return A.Compose([
            # A.OneOf([
                # A.Emboss(p=0.3),
            #     A.Sharpen(p=0.3),
            # ], p=0.5),
            # A.OneOf([
                # A.Blur(p=0.3),
                # A.GaussNoise(p=0.3, var_limit=(300, 400)),
                # A.MotionBlur(p=0.3, blur_limit=(10, 20)),
            # ], p=1),
            # A.Rotate(p=0.5, limit=[-35, 35]),
            valid_trans,
        ])

    return valid_trans

def get_loader(batch_size=4, num_workers=8, phase='train'):
    dataset = ActionDataset(
        transforms=get_transformer(phase=phase),
        num_frames=CFG.num_frames
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - test_size - val_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    if (phase == 'train'):
        dataset = train_dataset
    elif (phase == 'test'):
        dataset = test_dataset
    else:
        dataset = val_dataset

    return DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )