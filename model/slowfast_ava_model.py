import torch
from torch.nn import functional as F
from torch import nn
from torchmetrics.functional import accuracy
import pytorchvideo.models.slowfast as SlowFastModel
import torchvision.models as models
from data_preparation.action_dataset import ActionDataset
from data_preparation.actions import Action
from pytorch_lightning.core.module import LightningModule


class SlowFast(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=50):
        super().__init__()

        self.drop_prob = drop_prob
        self.num_frames = num_frames
        self.num_classes = len(Action().action)

        self.load()

    def load(self):
        self.slowfast = SlowFastModel.create_slowfast_with_roi_head(
            slowfast_channel_reduction_ratio=(8,),
            input_channels=(3, 3),
            model_depth=50,
            model_num_class=len(Action().action),  # Number of action classes
            dropout_rate=0.5,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            stem_function=(create_res_basic_stem, create_res_basic_stem),
            stem_dim_outs=(64, 8),
            # Other configurations as required...
        )

    def forward(self, video_frames, bboxes=None):
        out = self.slowfast(video_frames, bboxes)
        return out

    
    def configure_optimizers(self):
        learning_rate = 1e-4
        optimizer = torch.optim.ASGD(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=4, cooldown=2)
        return { "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss" }
    
    def on_training_epoch_end(self):
        sch = self.lr_schedulers()

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["valid_loss"])
        else:
            sch.step()

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        output = self(x)

        acc = accuracy(output, y,task="multiclass",num_classes=len(Action().action))
        loss = F.cross_entropy(output, y)
        metrics = {"train_acc": acc, "train_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"valid_acc": acc, "valid_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics
    
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        output = self(x)

        acc = accuracy(output, y,task="multiclass", num_classes=len(Action().action))
        loss = F.cross_entropy(output, y)

        return loss, acc
    
