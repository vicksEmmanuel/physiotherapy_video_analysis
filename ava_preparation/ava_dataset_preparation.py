from torch.utils.data import IterableDataset
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from data_preparation.actions import Action
from model.slowfast_model import SlowFast
from pytorchvideo.data import Ava
import pandas as pd
import json
from pytorchvideo.data import Ava
from pytorchvideo.data.clip_sampling import make_clip_sampler
import numpy as np
from torch.utils.data import DataLoader,random_split
from data_preparation.config import CFG
from data_preparation.util import ava_inference_transform2, ava_inference_transform, single_transformer, get_new_transformer
import os

from pytorchvideo.transforms.functional import (
    convert_to_one_hot,
)


def show_image(frame, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(frame)
    ax.axis('off')  # Hide the axis
    
    H, W = frame.shape[:2]  # Height and Width of the frame

    print(f"{H} {W}")

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        # Scale the box coordinates to match the image dimensions
        rect = patches.Rectangle((x_min * W, y_min * H), (x_max - x_min) * W, (y_max - y_min) * H,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show(block=False)
    plt.pause(.1)


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


def draw_image(sample_1):
    print(sample_1)
    frame = sample_1['video'][0] # Access the first video in the batch

    frame = frame[0, :, :].detach().cpu().numpy()
    frame = (frame - frame.min()) / (frame.max() - frame.min())
    boxes = sample_1['ori_boxes']  # Retrieve bounding box data
    show_image(frame, boxes)

def visualize_ava_dataset(dataset):
    # All videos are of the form cthw and fps is 30
    # Clip is samples at time step = 2 secs in video
    sample_1 = next(dataset)
    draw_image(sample_1)

def prepare_ava_dataset(phase='train', config=CFG):
    ava_frame_list = f"ava/frame_lists/{phase}.csv"
    df = pd.read_csv(ava_frame_list, sep=' ')
    json_array = df.to_dict(orient='records')
    json_string = json.dumps(json_array)

    ava_file_names_train_val = "ava/ava_file_names_trainval_v2.1.txt"
    with open(ava_file_names_train_val, 'r') as file:
        allFiles = file.read().splitlines()
        for i in range(0,5):
            continue


    label_map_path = "ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt"
    with open(label_map_path, 'r') as file:
        label_map = file.read()


    prepared_frame_list = f"ava_preparation/frame_lists/{phase}.csv"

    with open(prepared_frame_list, 'r') as file:
        allFiles = file.read().splitlines()
        for i in range(0, len(allFiles)):
            continue

    # @TODO: This is a temporary fix to avoid re-creating the file
    if not os.path.exists(prepared_frame_list):
        with open(prepared_frame_list, 'w') as prepared_frame_list_file:
            prepared_frame_list_file.write(f"original_vido_id video_id frame_id path labels\n")

            for i in range(0, len(json_array)):
                print(json_array[i])
                video_id = json_array[i]['original_vido_id']
                for file in allFiles:
                    if video_id in file:
                        prepared_frame_list_file.write(f"{video_id} {video_id} {json_array[i]['frame_id']} ava/frames/{json_array[i]['path']}" + " \"\" \n")



    frames_label_file_path = f"ava/annotations/ava_{phase}_v2.2.csv"

    def transform(sample_dict):
        return ava_inference_transform2(
            sample_dict, 
            num_frames = config.num_frames, 
            slow_fast_alpha = None,
            crop_size=256, 
            data_mean=[0.45, 0.45, 0.45], 
            data_std=[0.225, 0.225, 0.225]
        )


    prepared_frame_list = f"ava_preparation/real/frame_lists/{phase}.csv"
    frames_label_file_path = f"ava_preparation/real/annotations/{phase}.csv"


    iterable_dataset = Ava(
        frame_paths_file=prepared_frame_list,
        frame_labels_file=frames_label_file_path,
        clip_sampler=make_clip_sampler("random", 20.0),
        label_map_file=label_map_path,
        transform=transform
    )

    # dataset = AvaDataset(iterable_dataset)

    loader = DataLoader(
        iterable_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_batches_per_epoch=500,
        collate_fn=lambda x: x 
    )

    # Shows a picture of the first video in the dataset
    # visualize_ava_dataset(dataset)
    return loader



class AvaDataset(IterableDataset):
    def __init__(self, iterable_dataset):
        self.data = []
        for item in iterable_dataset:
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]