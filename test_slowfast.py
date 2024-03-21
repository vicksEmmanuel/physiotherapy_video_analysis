import os
from pytorchvideo.data.encoded_video import EncodedVideo
from data_preparation.actions import Action
import cv2
import torch
import numpy as np
from get_audio import get_audio
from model.slowfast_model import SlowFast  # Ensure this import matches your project structure
from torchvision import transforms
from data_preparation.config import CFG  # Ensure this import matches your project structure
from data_preparation.PackPathwayTransform import PackPathway
from data_preparation.util import get_transformer,get_new_transformer  # Ensure this import matches your project structure
from data_preparation.util_2 import  get_video_clip_and_resize # Ensure this import matches your project structure

# Assuming device setup as before
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_actions_from_video(video_path):
    new_path = get_video_clip_and_resize(video_path)
    video = EncodedVideo.from_path(new_path)

    total_duration = int(video.duration)  # Total duration in seconds

    audio_timestamps =  get_audio(video_path, total_duration=total_duration)

    # Load your trained SlowFast model
    model = SlowFast.load_from_checkpoint("checkpoints/last.ckpt")
    model.eval()
    model.to(device)

    transform = get_new_transformer('test')
    
    actions_per_second = []

    for i in range(0,len(audio_timestamps)):
        start_sec = audio_timestamps[i]["start"]
        end_sec = audio_timestamps[i]["end"]

        # Get the clip for the current second
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = transform(video_data)
        frames = video_data["video"]

        # Ensure frames are on the correct device
        frames = [i.to(device)[None, ...] for i in frames]

        confidence_threshold = 0.5

        with torch.no_grad():
            outputs = model(frames)
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(outputs)
            top_preds = preds.topk(k=3)
            pred_classes = top_preds.indices[0]
            confidences = top_preds.values[0]  # Get the confidence values of the top predictions

            actions_this_second = []
            for idx, confidence in enumerate(confidences):
                if confidence > confidence_threshold:
                    action_name = Action().action[int(pred_classes[idx])]
                    actions_this_second.append(action_name)
                else:
                    actions_this_second.append("")  # Placeholder for low confidence predictions

            # Log or use the actions_this_second as needed
            if actions_this_second:  # Check if the list is not empty
                print(f"Actions for second {start_sec}-{end_sec}: {actions_this_second}")
                actions_per_second.append({
                    "actions": actions_this_second,
                    "text": audio_timestamps[i]["text"],
                })
            else:
                print(f"No confident actions for second {start_sec}-{end_sec}.")

    return actions_per_second


video_path = 'videos/Postural Assessment.mp4'
all_actions = generate_actions_from_video(video_path)
print(all_actions)