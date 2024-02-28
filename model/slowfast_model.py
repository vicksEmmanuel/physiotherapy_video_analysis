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
        # Initialize the SlowFast model with the Kinetics number of classes first
        self.slowfast = SlowFastModel.create_slowfast(
            model_num_class=self.num_classes,
            dropout_rate=self.drop_prob,
        )
        # Load the pretrained weights for the Kinetics classes
        pretrained_dict = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True).state_dict()

        # Remove the final layer weights (since this will be adapted to the new number of classes)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'head.projection' not in k}

        # Update the model with the pretrained weights
        self.slowfast.load_state_dict(pretrained_dict, strict=False)

        # Replace the final layer with a new one that has the correct number of outputs
        num_features = self.slowfast.head.projection.in_features
        self.slowfast.head.projection = nn.Linear(num_features, self.num_classes)


    def forward(self, x):
        out = self.slowfast(x)
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
    
