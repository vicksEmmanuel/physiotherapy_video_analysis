from pytorch_lightning import Trainer, seed_everything
from data_preparation.action_dataset import ActionDataset
from data_preparation.config import CFG
from data_preparation.util import get_loader, get_transformer
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


def train(config):

    print("Training begins:")

    train = ActionDataset(
        transforms=get_transformer('valid'),
    )

    batch = train.__getitem__(10)

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
    

    # Plot the frame
    def show_image(frame):
        plt.imshow(frame)
        plt.axis('off')  # Hide the axis
        plt.show(block=False)
        plt.pause(.1)

    show_image(frame)

    loaders = {
        p: get_loader(config.batch_size, config.num_workers, p)
            for p in [ 'train', 'valid', 'test'] 
    }
    
    model = SlowFast(drop_prob=config.drop_prob)

    trainer = Trainer(
        # logger=wandb_logger,
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0,
        # overfit_batches=0.05,
        # callbacks=[lr_monitor],
    )

    trainer.fit(model, loaders['train'], loaders['valid'])
    trainer.test(model, loaders['test'])

train(CFG)
    