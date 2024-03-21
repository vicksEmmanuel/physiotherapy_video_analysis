from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning import Trainer, seed_everything
from data_preparation.action_dataset import ActionDataset
from data_preparation.config import CFG
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
from model.slowfast_ava_model import SlowFastAva
from model.slowfast_model import SlowFast
from ava_preparation.ava_dataset_preparation import prepare_ava_dataset
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths


def train(config):


    print("Training begins:")

    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt')
    action_ids = list(label_map.keys())
    print(f"Action IDs: {action_ids}")
    length_of_actions = len(action_ids)
    print(f"Length of actions:  {length_of_actions}")
    action_ids, length_of_actions

    print(f"Length of actions:  {length_of_actions}")


    model = SlowFastAva(
        drop_prob=config.drop_prob, 
        num_frames=config.num_frames,
        num_classes=length_of_actions
    )

    print(f"Model: {model}")

    loaders = {
        p: prepare_ava_dataset(p,  config=config)
            for p in [ 'train', 'val'] 
    }

   

    print("Loaders created")
    
    


    checkpoint_callback = ModelCheckpoint(
        dirpath='ava_checkpoints/', # Directory where the checkpoints will be saved
        filename='{epoch}-{val_loss:.2f}', # File name, which can include values from logging
        save_top_k=1, # Save the top 3 models according to the metric monitored below
        verbose=True,
        monitor='valid_loss', # Metric to monitor for improvement
        mode='min', # Mode 'min' is for loss, 'max' for accuracy
        every_n_epochs=1, # Save checkpoint every epoch
        save_last=True, # Save the last model regardless of the monitored metric
    )

    trainer = Trainer(
        # logger=wandb_logger,
        # accelerator='cpu', # 'ddp' for distributed computing
        # accelerator='gpu', # 'ddp' for distributed computing
        devices=1, # Use 1 GPU
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0,
        # overfit_batches=0.05,
        # callbacks=[lr_monitor],
        callbacks=[
            RichProgressBar(),
            checkpoint_callback
        ],
    )

    trainer.fit(model, loaders['train'], loaders['val'])
    # trainer.test(model, loaders['test'])

train(CFG)
    