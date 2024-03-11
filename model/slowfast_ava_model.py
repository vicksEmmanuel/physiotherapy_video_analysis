from pytorchvideo.models.hub import slow_r50_detection 
from pytorchvideo.models.resnet import create_resnet_with_roi_head
from torch import nn
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy

import torch
import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet_with_roi_head

# class CustomSlowFastAva(nn.Module):
#     def __init__(self, num_classes, drop_prob=0.5, input_channel=3, model_depth=50, norm=nn.BatchNorm3d, activation=nn.ReLU):
#         super().__init__()
#         self.base_model = create_resnet_with_roi_head(
#             model_num_class=num_classes,
#             dropout_rate=drop_prob,
#             input_channel=input_channel,
#             model_depth=model_depth,
#             norm=norm,
#             activation=activation,
#         )

#         # Add a temporal pooling layer to reduce the temporal dimension to 1
#         # Assuming the temporal dimension is at index 2 (BxCxTxHxW)
#         self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

#     def forward(self, x, bboxes):
#         # Apply temporal pooling
#         x = self.temporal_pool(x)
        
#         # Remove the singleton temporal dimension
#         x = x.squeeze(2)
        
#         return self.base_model(x, bboxes)


class CustomSlowR50Detection(nn.Module):
    def __init__(self, pretrained=True, num_classes=80):
        super().__init__()
        self.base_model = create_resnet_with_roi_head(
            model_num_class=num_classes
        )
        # detection_head = self.base_model.detection_head
        # num_features = detection_head.proj.in_features
        # detection_head.proj = nn.Linear(num_features, num_classes)


    def forward(self, x, bboxes):
        print(f"Videos shape: {x.shape} Bboxes shape: {bboxes.shape}")
        return self.base_model(x, bboxes)


class SlowFastAva(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=16, num_classes=5):
        super().__init__()

        self.drop_prob = drop_prob
        self.num_classes = 80 # num_classes
        self.num_frames = num_frames
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_hyperparameters()

        self.load()

    def load(self):
        self.model = CustomSlowR50Detection(pretrained=True, num_classes=self.num_classes)

    def forward(self, x, bboxes):
        print(f"Input shape before model: {x.shape}")  # Debug print
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

        total_loss = 0
        total_acc = 0
        for batch_item in batch:
            videos = batch_item['video']
            bboxes = batch_item['boxes']

            labels = batch_item['labels']
            labels = torch.tensor(labels, dtype=torch.long)
            new_label = torch.nn.functional.one_hot(labels, self.num_classes + 1)
            labels = new_label

            print(f"Videos shape: {videos.shape} Bboxes shape: {bboxes.shape}  Labels shape: {labels.shape}")
            outputs = self(videos.to(self.device), bboxes.to(self.device))

            loss = F.cross_entropy(outputs, labels)
            acc = accuracy(outputs.softmax(dim=-1), labels, num_classes=self.num_classes)

            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / len(batch)
        avg_acc = total_acc / len(batch)

        metrics = {"train_acc": avg_acc, "train_loss": avg_loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        print("Training step")

        total_loss = 0
        total_acc = 0
        for batch_item in batch:
            videos = batch_item['video']
            bboxes = batch_item['boxes']

            labels = batch_item['labels']
            labels = torch.tensor(labels, dtype=torch.long)
            new_label = torch.nn.functional.one_hot(labels, self.num_classes + 1)
            labels = new_label

            print(f"Videos shape: {videos.shape} Bboxes shape: {bboxes.shape}  Labels shape: {labels.shape}")
            outputs = self(videos.to(self.device), bboxes.to(self.device))


            loss = F.cross_entropy(outputs, labels)
            acc = accuracy(outputs.softmax(dim=-1), labels, num_classes=self.num_classes)

            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / len(batch)
        avg_acc = total_acc / len(batch)

        metrics = {"valid_acc": avg_acc, "valid_loss": avg_loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):

        total_loss = 0
        total_acc = 0
        for batch_item in batch:
            videos = batch_item['video']
            bboxes = batch_item['boxes']
            
            labels = batch_item['labels']
            labels = torch.tensor(labels, dtype=torch.long)
            new_label = torch.nn.functional.one_hot(labels, self.num_classes + 1)
            labels = new_label
            
            print(f"Videos shape: {videos.shape} Bboxes shape: {bboxes.shape}  Labels shape: {labels.shape}")
            outputs = self(videos.to(self.device), bboxes.to(self.device))

            loss = F.cross_entropy(outputs, labels)
            acc = accuracy(outputs.softmax(dim=-1), labels, num_classes=self.num_classes)

            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / len(batch)
        avg_acc = total_acc / len(batch)
        
        metrics = {"test_acc": avg_acc, "test_loss": avg_loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics
