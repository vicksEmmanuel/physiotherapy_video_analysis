import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import torch
from torchvision import transforms
from data_preparation.PackPathwayTransform import PackPathway
from torch.utils.data import Dataset

from data_preparation.actions import Action


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


class ActionDataset(Dataset):
    def __init__(self, transforms=None, num_frames=200, data_path='data_preparation/actions'):
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
        return self.actions.index(idx_action)
    
    def __getitem__(self, idx):
       path = self.all_videos[idx][0]
       frames = []
       cap = cv2.VideoCapture(path) # Get the video path
       v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get the video length
       frame_idx = np.sort(np.random.choice(np.arange(v_len-1), self.num_frames))   #   get random frame indices, sometimes the last frame generates an error, therefore v_len-1

       #   iterate for each frame
       for i in frame_idx:
            img = torch.zeros((3, 224, 224))    #   empty tensor in case frame will not read by cv2
            cap.set(cv2.CAP_PROP_POS_FRAMES, i) #   move to relevant frame index
            ret, frame = cap.read()

            # if frame was read
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                if self.transforms:   
                    img = self.transforms(image=img)['image']
            frames.append(img)
       
       cap.release()   #   release video

       labels = self.convert_action_to_numpy(self.all_videos[idx][1])
       frames = torch.stack(frames)

       frames = torch.permute(frames, (1, 0, 2, 3))
       frames = self.pack_pathway(frames)

       return frames, labels, idx, dict()
    
    
