from pytorchvideo.models.resnet import create_resnet_with_roi_head
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorchvideo.models.slowfast import create_slowfast, create_slowfast_with_roi_head
from pytorchvideo.models.hub import slow_r50_detection 

import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy

class SlowFastAva(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=16, num_classes=5):
        super().__init__()

        self.drop_prob = drop_prob
        self.num_classes = 80 # num_classes
        self.num_frames = num_frames
        self.save_hyperparameters()

        self.load()

    def load(self):
        self.model = slow_r50_detection(
            pretrained=False,
            model_num_class=self.num_classes,
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
        selected_batch = batch[batch_idx]
        video = selected_batch["video"]
        boxes = selected_batch["boxes"]
        labels = selected_batch["labels"]
        labels = torch.tensor(labels, dtype=torch.long)

        one_hot_labels = torch.zeros((len(labels), self.num_classes))
        for i, label_list in enumerate(labels):
            for label in label_list:
                one_hot_labels[i, label - 1] = 1

        labels = one_hot_labels

        print(f"Video shape: {video.shape} and boxes shape: {boxes.shape} and labels shape: {labels}")

        outputs = self.model(video, boxes)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs.softmax(dim=-1), labels,task="multiclass",num_classes=self.num_classes)
        metrics = {"train_acc": acc, "train_loss": loss}
        
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        selected_batch = batch[batch_idx]
        video = selected_batch["video"]
        boxes = selected_batch["boxes"]
        labels = selected_batch["labels"]
        labels = torch.tensor(labels, dtype=torch.long)

        one_hot_labels = torch.zeros((len(labels), self.num_classes))
        for i, label_list in enumerate(labels):
            for label in label_list:
                one_hot_labels[i, label - 1] = 1

        labels = one_hot_labels

        outputs = self.model(video, boxes)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs.softmax(dim=-1), labels,task="multiclass",num_classes=self.num_classes)

        metrics = {"valid_acc": acc, "valid_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        selected_batch = batch[batch_idx]
        video = selected_batch["video"]
        boxes = selected_batch["boxes"]
        labels = selected_batch["labels"]
        labels = torch.tensor(labels, dtype=torch.long)

        one_hot_labels = torch.zeros((len(labels), self.num_classes))
        for i, label_list in enumerate(labels):
            for label in label_list:
                one_hot_labels[i, label - 1] = 1

        labels = one_hot_labels


        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs.softmax(dim=-1), labels,task="multiclass",num_classes=self.num_classes)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics
