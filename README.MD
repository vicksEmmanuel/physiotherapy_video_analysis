<!-- @format -->

## Installation

```
pip install -r requirement.txt
```

Make sure to download `ffmpeg` on your device

```
python data-preparation/create_action.py --input "Brief screenning.MOV" --action "moving backwards" --start "00:00:39" --end "00:00:49"
```

Where
`   --input should be a video already in the videos folder
    --action should be the action discovered in the input video
    --start is the time the action begins
    --end is the time the action ends
  `
