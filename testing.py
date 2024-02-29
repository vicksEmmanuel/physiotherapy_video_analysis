from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from data_preparation.action_dataset import ActionDataset
from data_preparation.config import CFG
from data_preparation.util import get_loader, get_transformer,get_new_transformer
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
from model.slowfast_model import SlowFast

print(Action().action)


   # Plot the frame
def show_image(frame):
    plt.imshow(frame)
    plt.axis('off')  # Hide the axis
    plt.show(block=False)
    plt.pause(.1)

train = ActionDataset(
        transforms=get_new_transformer('train'),
        num_frames=CFG.num_frames
    )

print(train)
batch = train.__getitem__(10)
print(batch)

frames = batch[0]
first_video_frames = frames[0]  # Access the first tensor in the list

num_frames = first_video_frames.shape[1]

# Select the first frame for visualization
# Assuming the tensor has dimensions [C, F, H, W], select the first frame
frame = first_video_frames[:, 0, :, :].detach().cpu().numpy()  # Convert to numpy

# Transpose the frame from [C, H, W] to [H, W, C] for plotting
frame = np.transpose(frame, (1, 2, 0))

# Normalize the frame's pixel values to [0, 1] for correct visualization
frame = (frame - frame.min()) / (frame.max() - frame.min())

show_image(frame)

# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
#         test_acc            0.20000000298023224
#         test_loss            5.677567958831787