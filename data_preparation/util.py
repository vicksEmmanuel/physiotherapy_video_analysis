import random
import numpy as np
import torch
from torch.utils.data import DataLoader,random_split
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

    # if phase == 'train':
    #     return A.Compose([
    #         # A.OneOf([
    #             # A.Emboss(p=0.3),
    #         #     A.Sharpen(p=0.3),
    #         # ], p=0.5),
    #         # A.OneOf([
    #             # A.Blur(p=0.3),
    #             # A.GaussNoise(p=0.3, var_limit=(300, 400)),
    #             # A.MotionBlur(p=0.3, blur_limit=(10, 20)),
    #         # ], p=1),
    #         # A.Rotate(p=0.5, limit=[-35, 35]),
    #         valid_trans,
    #     ])

    return valid_trans

def get_loader(batch_size=4, num_workers=8, phase='train'):
    dataset = ActionDataset(
        transforms=get_transformer(phase=phase),
        num_frames=CFG.num_frames
    )

    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(0.1 * total_size)
    val_size = int(0.1 * total_size)
    train_size = total_size - test_size - val_size

    # Split dataset into training, validation, and test sets
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    # Select the appropriate dataset for the phase
    if phase == 'train':
        selected_dataset = train_dataset
        shuffle = True  # Typically you shuffle the training dataset
    elif phase == 'test':
        selected_dataset = test_dataset
        shuffle = False  # No need to shuffle the test dataset
    else:  # 'val' or any other phase defaults to validation
        selected_dataset = val_dataset
        shuffle = False  # No need to shuffle the validation dataset

    # Create the DataLoader for the selected dataset
    loader = DataLoader(
        selected_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return loader
