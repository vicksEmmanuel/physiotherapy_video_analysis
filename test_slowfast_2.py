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


def crop_and_transform_frame(frame, box, size=256):
    # Ensure box coordinates are integers
    top, left, bottom, right = map(int, box)
    # Calculate height and width from the bounding box
    height = bottom - top
    width = right - left
    # Crop the frame using the integer coordinates
    # Check if frame is a PIL Image or already a tensor
    if isinstance(frame, torch.Tensor):
        # Assume frame is in CHW format
        cropped_frame = frame[:, top:top+height, left:left+width]
    else:
        cropped_frame = F.crop(frame, top, left, height, width)

    # Resize the cropped frame
    # If cropped_frame is already a tensor, use torchvision.transforms.functional.resize
    # Else, it's a PIL Image, use F.resize as before
    if isinstance(cropped_frame, torch.Tensor):
        resized_frame = F.resize(cropped_frame, [size, size], antialias=True)
    else:
        resized_frame = F.resize(cropped_frame, [size, size], antialias=True)
    
    # If the frame is already a tensor, normalize directly
    if isinstance(resized_frame, torch.Tensor):
        normalized_frame = F.normalize(resized_frame, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    else:
        # Convert PIL image to tensor and normalize
        tensor_frame = F.to_tensor(resized_frame)
        normalized_frame = F.normalize(tensor_frame, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    
    return normalized_frame

def crop_tensor_frame(frame, box):
    """
    Crop a tensor frame to the specified bounding box.
    
    Parameters:
    - frame: a tensor representing an image, expected shape [C, H, W]
    - box: the bounding box coordinates (x1, y1, x2, y2)
    
    Returns:
    - Cropped frame as a tensor.
    """
    # Convert box coordinates to integer pixel indices
    x1, y1, x2, y2 = map(int, box)
    # Crop the frame using tensor slicing
    cropped_frame = frame[:, y1:y2, x1:x2]
    return cropped_frame


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

    # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
    predicted_boxes = get_person_bboxes(inp_img, predictor)
    if len(predicted_boxes) == 0:
        print(f"No person detected in second {start_sec} - {end_sec}.")
        continue


    
    for box in predicted_boxes:
        cropped_frames = [crop_tensor_frame(frame, box) for frame in inp_imgs]
        transformer = single_transformer()
        transformed_clips = [transformer(frame.unsqueeze(0)) for frame in cropped_frames]  # Add dummy batch dimension

        
        new_transformed_clips = [i.to(device)[None, ...] for i in transformed_clips]
        roi_clip_tensor = torch.stack(new_transformed_clips, dim=0).unsqueeze(0).to(device) # Add batch dim

        print(roi_clip_tensor)

        with torch.no_grad():
            outputs = model(roi_clip_tensor)
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(outputs)
            top_preds = preds.topk(k=1)
            pred_classes = top_preds.indices[0]
            confidences = top_preds.values[0]

            actions_this_second = []
            for idx, confidence in enumerate(confidences):
                if confidence > 0.5:
                    action_name = Action().action[int(pred_classes[idx])]
                    actions_this_second.append(action_name)
                else:
                    actions_this_second.append("")  # Placeholder for low confidence predictions

            print(f"Actions for second {start_sec}-{end_sec}: {actions_this_second}")
    
        # # Annotate frame with action labels and bounding boxes
        # annotated_frame = annotate_frame(frame, box, action_label) # Implement this
        
        # out.write(annotated_frame)  # Write frame to output video

print("Finished generating predictions.")