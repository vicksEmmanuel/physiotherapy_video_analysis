from PIL import Image, ImageDraw
import os
from torchvision.transforms.functional import normalize as normalize_image
from torchvision.transforms.functional import to_tensor
from data_preparation.actions import Action
from model.slowfast_model import SlowFast  # Ensure this import matches your project structure
from detectron2.config import get_cfg
from pytorchvideo.models.hub import slow_r50_detection 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
import torch
from torch.nn import functional as F
from torch import nn
from torchmetrics.functional import accuracy
from torchvision.transforms._functional_video import normalize
import numpy as np
from data_preparation.util_2 import  get_video_clip_and_resize # Ensure this import matches your project structure
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
from torchvision.transforms import functional as F
from detectron2.utils.video_visualizer import VideoVisualizer
import pytorchvideo.models.slowfast as SlowFastModel
import cv2
from model.slowfast_ava_model import SlowFastAva  # Ensure this import matches your project structure
from data_preparation.util import single_transformer
from pytorchvideo.models.resnet import create_resnet, create_resnet_with_roi_head
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths


video_path = 'data_preparation/actions/pelvis check/2024-02-14 12-46-31.mp4'
new_path = get_video_clip_and_resize(video_path)
encoded_vid = EncodedVideo.from_path(new_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = slow_r50_detection(True)
print(model)

model.eval()
model.to(device)

print("Model loaded.")

actions = Action().action
label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
video_visualizer = VideoVisualizer(80, label_map, top_k=3, mode="thres",thres=0.5)

print(f"Video action loaded. {actions}")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes

def ava_inference_transform(
    clip,
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (256, 256))

gif_imgs = []
total_duration = int(encoded_vid.duration)  # Total duration in seconds

for start_sec in range(0, total_duration):
    end_sec = start_sec + 5  # Process one second at a time

    # Generate clip around the designated time stamps
    inp_imgs = encoded_vid.get_clip(start_sec=start_sec, end_sec=end_sec)
    inp_imgs = inp_imgs['video']
    

    # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
    # We use the the middle image in each clip to generate the bounding boxes.
    inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
    inp_img = inp_img.permute(1,2,0)

    # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
    predicted_boxes = get_person_bboxes(inp_img, predictor)
    if len(predicted_boxes) == 0:
        print(f"No person detected in second {start_sec} - {end_sec}.")
        continue

    print(f"Predicted boxes for second {start_sec} - {end_sec}: {predicted_boxes.numpy()}")

    inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())

    print(f"Inputs shape: {inputs.shape}")
    print(f"Bounding boxes shape: {inp_boxes.shape}")

    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
    inputs = inputs.unsqueeze(0)

    print(f"Inputs shape: {inputs.shape}")
    print(f"Bounding boxes shape: {inp_boxes.shape}")

    # Generate actions predictions for the bounding boxes in the clip.
    # The model here takes in the pre-processed video clip and the detected bounding boxes.
    preds = model(inputs.to(device), inp_boxes.to(device))

    # print(f"Predictions for second {start_sec} - {end_sec}: {preds}")

    preds= preds.to('cpu')
    # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
    preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)

    # Plot predictions on the video and save for later visualization.
    inp_imgs = inp_imgs.permute(1,2,3,0)
    inp_imgs = inp_imgs/255.0
    out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
    gif_imgs += out_img_pred


    
    
   

print("Finished generating predictions.")


height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]

vide_save_path = 'output.mp4'
video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'DIVX'), 7, (width,height))

for image in gif_imgs:
    img = (255*image).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    video.write(img)
video.release()

print('Predictions are saved to the video file: ', vide_save_path)