import whisper
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from data_preparation.action_dataset import ActionDataset
from data_preparation.config import CFG
from data_preparation.util import get_loader, get_transformer,get_new_transformer
import torch
from ava_preparation.ava_dataset_preparation import prepare_ava_dataset
import subprocess
import azure.cognitiveservices.speech as speechsdk
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_video_to_audio_file(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vn',
        '-acodec', 'pcm_s16le',  # Linear PCM format
        '-ar', '16000',  # Set audio sampling rate to 16000 Hz
        '-ac', '1',  # Set audio channels to mono
        output_file
    ]
    subprocess.run(command)

# Function to insert silent segments
def insert_silent_segments(segments, total_duration):
    updated_segments = []
    current_time = 0.0

    for segment in segments:
        gap_duration = segment["start"] - current_time
        # If there's a gap between the current time and the start of the next segment
        if gap_duration > 1:
            # Add a silent segment
            updated_segments.append({
                "id": len(updated_segments),
                "start": current_time,
                "end": segment["start"],
                "text": ""
            })
            # Add the actual segment
            updated_segments.append({
                "id": len(updated_segments),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        elif gap_duration <= 1:
            # If the gap is less than or equal to 1 second, merge with the previous or next segment
            if updated_segments:
                # Adjust the end time of the previous segment if it exists
                updated_segments[-1]["end"] = segment["start"]
            # If no segments have been added yet, treat it as starting at 0
            else:
                current_time = 0
            # Add the current segment
            updated_segments.append({
                "id": len(updated_segments),
                "start": current_time,
                "end": segment["end"],
                "text": segment["text"]
            })

        # Update the current time
        current_time = segment["end"]

    # Check if there's still time left after the last segment till the total duration
    if current_time < total_duration:
        gap_duration = total_duration - current_time
        if gap_duration > 1:
            updated_segments.append({
                "id": len(updated_segments),
                "start": current_time,
                "end": total_duration,
                "text": ""
            })
        else:
            # Adjust the end time of the last segment if the gap is less than or equal to 1 second
            if updated_segments:
                updated_segments[-1]["end"] = total_duration

    return updated_segments

def get_audio(video_path, total_duration):

    SPEECH_FILE = 'temporary_audio.wav'
    convert_video_to_audio_file(video_path, SPEECH_FILE)

    # Load the model
    model = whisper.load_model("base")

    # Process the audio
    result = model.transcribe(SPEECH_FILE)
    segments = result['segments']

    # Calling the function to insert silent segments
    updated_segments = insert_silent_segments(segments, total_duration)

    print(updated_segments)

    print("Deleting the audio file")
    os.remove(SPEECH_FILE)
    return updated_segments

