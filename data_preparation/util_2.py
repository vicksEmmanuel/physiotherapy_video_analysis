from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
from torch.utils.data import DataLoader,random_split
import albumentations as A
import cv2
from data_preparation.config import *



def get_video_clip_and_resize(video_path, start_sec, end_sec, target_size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Ensure start_sec and end_sec are within the video's duration
    start_sec = max(0, min(start_sec, duration))
    end_sec = max(0, min(end_sec, duration))
    
    # Calculate start and end frames
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # Initialize a list to hold resized frames
    frames = []
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frames within the desired segment
        if start_frame <= current_frame <= end_frame:
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            frames.append(resized_frame)
        
        current_frame += 1
        if current_frame > end_frame:
            break
    
    cap.release()
    
    # Convert the list of frames to a numpy array
    clip = np.stack(frames)
    return clip
