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
from functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
import numpy as np
from data_preparation.util_2 import  get_video_clip_and_resize # Ensure this import matches your project structure
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
from torchvision.transforms import functional as F
from detectron2.utils.video_visualizer import VideoVisualizer
import cv2

from data_preparation.util import single_transformer


video_path = 'data_preparation/actions/pelvis check/2024-02-14 12-46-31.mp4'
new_path = get_video_clip_and_resize(video_path)
encoded_vid = EncodedVideo.from_path(new_path)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SlowFast.load_from_checkpoint("checkpoints/last.ckpt")
model.eval()
model.to(device)

print("Model loaded.")

actions = Action().action
label_map = {i: actions[i] for i in range(0, len(actions))}
# video_visualizer = VideoVisualizer(len(Action().action), label_map, top_k=3, mode="thres",thres=0.5)

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
    end_sec = start_sec + 1  # Process one second at a time

    # Generate clip around the designated time stamps
    inp_imgs = encoded_vid.get_clip(start_sec=start_sec, end_sec=end_sec)
    inp_imgs = inp_imgs['video']
    

    # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
    # We use the the middle image in each clip to generate the bounding boxes.
    inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
    inp_img = inp_img.permute(1,2,0)

    inp_img_pil = Image.fromarray(inp_img.byte().numpy())  # Convert to PIL Image
    draw = ImageDraw.Draw(inp_img_pil) 

    # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
    predicted_boxes = get_person_bboxes(inp_img, predictor)
    if len(predicted_boxes) == 0:
        print(f"No person detected in second {start_sec} - {end_sec}.")
        continue


    for box in predicted_boxes:
        # Convert tensor to list for drawing
        box = box.tolist()
        draw.rectangle(box, outline="red", width=3)
    
    # Display the image with the drawn bounding boxes
    # inp_img_pil.show()
    output_path = os.path.join("output_dir", f"frame_{start_sec}_{end_sec}.png")
    inp_img_pil.save(output_path)

    inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())
    frames = [i.to(device)[None, ...] for i in inputs]


    
    
   

print("Finished generating predictions.")