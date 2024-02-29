from pytorchvideo.data.encoded_video import EncodedVideo
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import torch
from torchvision import transforms
from data_preparation.config import CFG
from data_preparation.PackPathwayTransform import PackPathway
from torch.utils.data import Dataset

from data_preparation.actions import Action


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


class ActionDataset(Dataset):
    def __init__(self, transforms=None, num_frames=50, data_path='data_preparation/actions'):
        self.transforms = transforms
        self.num_frames = num_frames
        self.pack_pathway = PackPathway()
        self.data_path = data_path
        self.actions = Action().action
        self.all_videos = self.get_actions_video()


    def get_actions_video(self):
        video_paths = []

        data_path = self.data_path
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name) # Get the path to the class

            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    file_path = os.path.join(class_path, file)
                    video_paths.append((file_path, class_name))
        return video_paths
                    
    def __len__(self):
        return len(self.all_videos)
    
    def convert_action_to_numpy(self, idx_action):
        return int(self.actions.index(idx_action))
    
    def __getitem__(self, idx):
        path = self.all_videos[idx][0]
        print(path)
        frames = []
        cap = cv2.VideoCapture(path)  # Get the video path
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the video length
        clip_duration = v_len / cap.get(cv2.CAP_PROP_FPS)

        start_sec = 0
        end_sec = start_sec + clip_duration
        video = EncodedVideo.from_path(path)
        labels = self.convert_action_to_numpy(self.all_videos[idx][1])

        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        # Apply a transform to normalize the video input
        video_data = self.transforms(video_data)
        # Move the inputs to the desired device
        frames = video_data["video"]

        return frames, labels, idx, dict()

    
    
