import torch
import torch.nn as nn
from pytorchvideo.models.head import create_res_roi_pooling_head
from pytorchvideo.models.resnet import create_bottleneck_block, create_res_basic_stem
from pytorchvideo.models.resnet import create_resnet, create_resnet_with_roi_head

class DetectionBBoxNetwork(nn.Module):
    def __init__(self, num_classes):
        super(DetectionBBoxNetwork, self).__init__()
        # Define the stem
        self.stem = create_res_basic_stem(
            input_channels=3,
            output_channels=64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3)
        )
        
        # Define stages with custom or pre-built bottleneck blocks
        # Example for one stage; repeat and adjust for others
        self.stage1 = nn.Sequential(
            create_bottleneck_block(
                in_channels=64,
                out_channels=256,
                bottleneck_channels=64,
                stride=1,
                norm_layer=nn.BatchNorm3d,
                activation_layer=nn.ReLU
            ),
            # Add more blocks as needed
        )
        
        # Define the detection head (ROI pooling head here as an example)
        self.detection_head = create_res_roi_pooling_head(
            in_features=2048,  # Adjust based on your architecture
            out_features=num_classes,
            pool_kernel_size=(4, 1, 1),
            resolution=(7, 7),
            spatial_scale=0.0625,
            sampling_ratio=0
        )

    def forward(self, x, bboxes=None):
        x = self.stem(x)
        x = self.detection_head(x, bboxes)
        return x

# Instantiate the model
num_classes = 80
model = DetectionBBoxNetwork(num_classes=num_classes)
print(model)