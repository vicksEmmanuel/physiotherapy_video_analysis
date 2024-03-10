from torchvision.transforms.functional import resize
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.transforms._functional_video import normalize
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
import numpy as np
import torch
from torchvision.transforms._functional_video import normalize
from torch.utils.data import DataLoader,random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 


import cv2
from data_preparation.PackPathwayTransform import PackPathway
from data_preparation.config import *
from data_preparation.action_dataset import ActionDataset



def get_new_transformer(phase):
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
    slowfast_alpha = 4
    num_clips = 10
    num_crops = 3

    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        ),
    )

    return transform


def single_transformer():
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    return Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        )

def preprocess_transformer(roi):
    """
    Preprocess the ROI for action recognition model input.
    """
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(roi).unsqueeze(0)  # Add batch dimension

def ava_inference_transform(
    clip,
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes

def adjust_boxes(boxes, original_height, original_width, new_height, new_width):
    # Calculate scale factors for width and height
    width_scale = new_width / original_width
    height_scale = new_height / original_height

    # Adjust box coordinates
    adjusted_boxes = boxes.copy()
    adjusted_boxes[:, 0] = boxes[:, 0] * width_scale  # x1
    adjusted_boxes[:, 2] = boxes[:, 2] * width_scale  # x2
    adjusted_boxes[:, 1] = boxes[:, 1] * height_scale  # y1
    adjusted_boxes[:, 3] = boxes[:, 3] * height_scale  # y2

    return adjusted_boxes

def ava_inference_transform2(sample_dict, num_frames=4, crop_size=256, data_mean=[0.45, 0.45, 0.45], data_std=[0.225, 0.225, 0.225], slow_fast_alpha=None):
    clip = sample_dict["video"]
    boxes = np.array(sample_dict.get("boxes", []))
    ori_boxes = boxes.copy()


    clip = resize(clip, (crop_size, crop_size))
    boxes = adjust_boxes(boxes, clip.shape[2], clip.shape[3], crop_size, crop_size)

   # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)
    # boxes = torch.cat([torch.zeros(boxes.shape[0],1), boxes], dim=1)


    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    num_boxes = boxes.shape[0]
    dummy_labels = np.zeros((num_boxes, 1))
    boxes_with_labels = np.hstack((boxes, dummy_labels))

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]
    

    # boxes = torch.cat([torch.zeros(boxes.shape[0],1), boxes], dim=1)


    # Update sample_dict with transformed data
    transformed_sample_dict = sample_dict.copy()
    transformed_sample_dict["video"] = clip.unsqueeze(0)

    # label = torch.tensor(sample_dict["labels"])
    # new_label = torch.nn.functional.one_hot(label, 81)
    # transformed_sample_dict["labels"] = new_label
    

    if len(boxes) > 0:
        transformed_sample_dict["boxes"] = torch.from_numpy(boxes_with_labels).float()
    transformed_sample_dict["ori_boxes"] = torch.from_numpy(boxes).float()

    return transformed_sample_dict


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
        transforms=get_new_transformer(phase=phase),
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
