import argparse
import ffmpeg
import os
import datetime

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start', help='Start time in HH:MM:SS format')
parser.add_argument('--end', help='End time in HH:MM:SS format')
parser.add_argument('--input', help='Video file to get the action trimmed out of, this video should already be in the video folder')
parser.add_argument('--action', help='Action to trim out of the video')
args = parser.parse_args()

# Prepare file paths
video_to_get_actions_from = 'videos/' + args.input
video_name = os.path.basename(args.input)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
output_dir = f"data-preparation/actions/{args.action}"
output_file = f"{output_dir}/{timestamp}.mp4"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(video_name, args.start, args.end, args.action, video_to_get_actions_from, output_file)


(ffmpeg.input(video_to_get_actions_from, ss=args.start, to=args.end)
	.output(output_file)
	.run())