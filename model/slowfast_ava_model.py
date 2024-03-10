from pytorchvideo.models.resnet import create_resnet_with_roi_head
from torch import nn
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy

import torch
import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet_with_roi_head

class CustomSlowFastAva(nn.Module):
    def __init__(self, num_classes, drop_prob=0.5, input_channel=3, model_depth=50, norm=nn.BatchNorm3d, activation=nn.ReLU):
        super().__init__()
        self.base_model = create_resnet_with_roi_head(
            model_num_class=num_classes,
            dropout_rate=drop_prob,
            input_channel=input_channel,
            model_depth=model_depth,
            norm=norm,
            activation=activation,
        )

        # Add a temporal pooling layer to reduce the temporal dimension to 1
        # Assuming the temporal dimension is at index 2 (BxCxTxHxW)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, x, bboxes):
        # Apply temporal pooling
        x = self.temporal_pool(x)
        
        # Remove the singleton temporal dimension
        x = x.squeeze(2)
        
        return self.base_model(x, bboxes)


class SlowFastAva(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=16, num_classes=5):
        super().__init__()

        self.drop_prob = drop_prob
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.save_hyperparameters()

        self.load()

    def load(self):
        self.model = CustomSlowFastAva(
            num_classes=self.num_classes,
            drop_prob=self.drop_prob,
        )

    def forward(self, x, bboxes):
        return self.model(x, bboxes)

    def configure_optimizers(self):
        learning_rate = 1e-4
        optimizer = torch.optim.ASGD(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=4, cooldown=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}
    
    def on_training_epoch_end(self):
        sch = self.lr_schedulers()

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["valid_loss"])

    def training_step(self, batch, batch_idx):
        print("Training step")
        print(batch)
        videos = batch['videos'] 
        bboxes = batch['boxes'] 
        labels = batch['labels']
        outputs = self(videos, bboxes)

        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs.softmax(dim=-1), labels, num_classes=self.num_classes)

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        videos = batch['videos'] 
        bboxes = batch['boxes'] 
        labels = batch['labels']
        outputs = self(videos, bboxes)

        outputs = self(videos, bboxes)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs.softmax(dim=-1), labels, num_classes=self.num_classes)

        metrics = {"valid_acc": acc, "valid_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        videos = batch['videos'] 
        bboxes = batch['boxes'] 
        labels = batch['labels']

        outputs = self(videos, bboxes)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs.softmax(dim=-1), labels, num_classes=self.num_classes)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics
