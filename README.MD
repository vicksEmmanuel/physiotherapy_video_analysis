<!-- @format -->

## Installation

```
pip install -r requirement.txt
```

Make sure to download `ffmpeg` on your device

### Import Video Data

```
python videos/get_videos.py
```

After importing video data completes, you can then extract different actions with the command below

```
python data_preparation/create_action.py --input "Brief screenning.MOV" --action "moving backwards" --start "00:00:39" --end "00:00:49"
```

Where
`--input should be a video already in the videos folder
    --action should be the action discovered in the input video
    --start is the time the action begins
    --end is the time the action ends`

## Building Actions Dataset

```
python data_preparation/action_dataset.py
```


## Train Actions Dataset

```
python train_slowfast.py
```

### You'd get results like this

```
# 500 epochs

# Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████| 3/3 [00:12<00:00,  0.23it/s]
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#         test_acc             0.800000011920929
#         test_loss           0.6259332895278931
# ──────────────────────────────────────────────────────────────
```


## Test Action Dataset

```
video_path = 'videos/Postural Analysis Lateral.mp4'
all_actions = generate_actions_from_video(video_path)
print(all_actions)

python3 test_slowfast.py
```
