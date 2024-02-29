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


train = ActionDataset(
        transforms=get_new_transformer('train'),
        num_frames=CFG.num_frames
    )


# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
#         test_acc            0.20000000298023224
#         test_loss            5.677567958831787