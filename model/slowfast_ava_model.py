from pytorchvideo.models.resnet import create_resnet_with_roi_head
from torch import nn
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
import torch.nn as nn
from torchmetrics.functional import accuracy

class SlowFastAva(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=16, num_classes=5):
        super().__init__()

        self.drop_prob = drop_prob
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.save_hyperparameters()

        self.load()

    def load(self):
        self.model = create_resnet_with_roi_head(
            model_num_class=self.num_classes,
            dropout_rate=self.drop_prob,
            input_channel=3,
            model_depth=50,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
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
        x, y = batch[0], batch[1]
        
        output = self(x['video'], y['bboxes'])

        acc = accuracy(output, y,task="multiclass",num_classes=self.num_classes)
        loss = F.cross_entropy(output, y)
        metrics = {"train_acc": acc, "train_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        
        output = self(x['video'], y['bboxes'])

        acc = accuracy(output, y,task="multiclass",num_classes=self.num_classes)
        loss = F.cross_entropy(output, y)
        metrics = {"train_acc": acc, "train_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        videos, bboxes, labels = batch
        outputs = self(videos, bboxes)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs.softmax(dim=-1), labels, num_classes=self.num_classes)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics
