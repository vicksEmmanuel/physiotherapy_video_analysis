from pytorch_lightning.callbacks import ModelCheckpoint
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
        transforms=get_transformer('train'),
        num_frames=config.num_frames
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

    print("Loaders created")
    
    model = SlowFast(drop_prob=config.drop_prob, num_frames=config.num_frames)

    print(f"Model created: {model}")

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/', # Directory where the checkpoints will be saved
        filename='{epoch}-{val_loss:.2f}', # File name, which can include values from logging
        save_top_k=3, # Save the top 3 models according to the metric monitored below
        verbose=True,
        monitor='valid_loss', # Metric to monitor for improvement
        mode='min', # Mode 'min' is for loss, 'max' for accuracy
        every_n_epochs=1, # Save checkpoint every epoch
        save_last=True, # Save the last model regardless of the monitored metric
    )

    

    trainer = Trainer(
        # logger=wandb_logger,
        # accelerator='cpu', # 'ddp' for distributed computing
        accelerator='gpu', # 'ddp' for distributed computing
        devices=1, # Use 1 GPU
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0,
        # overfit_batches=0.05,
        # callbacks=[lr_monitor],
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, loaders['train'], loaders['valid'])
    trainer.test(model, loaders['test'])

train(CFG)
    