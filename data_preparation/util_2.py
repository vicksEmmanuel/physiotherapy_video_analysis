import os
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
from torch.utils.data import DataLoader,random_split
import albumentations as A
import cv2
from data_preparation.config import *



def get_video_clip_and_resize(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    path_without_extension = os.path.splitext(video_path)[0]
    new_path = f"{path_without_extension}_resized.mp4"
    out = cv2.VideoWriter(new_path, fourcc, 20.0, (256, 256))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame
        resized_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        # Write the resized frame to the new video
        out.write(resized_frame)

    # Release everything when job is finished
    cap.release()
    out.release()

    return new_path
