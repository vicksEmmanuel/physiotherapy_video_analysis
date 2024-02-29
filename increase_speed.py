import datetime
import subprocess

# def speed_up_video(input_file, output_file, speed_factor):
#     command = [
#         'ffmpeg',
#         '-i', input_file,
#         '-filter_complex',
#         f"[0:v]setpts=PTS/{speed_factor}[v];[0:a]atempo={speed_factor}[a]",
#         '-map', '[v]',
#         '-map', '[a]',
#         output_file
#     ]
#     subprocess.run(command)

# # Example usage
# input_file = 'RPReplay_Final1709231743.MP4'
# output_file = 'WhatsApp Video 2024-02-29 at 19.03.00 _speed_up.mp4'
# speed_factor = 1.5  # Increase speed by 1.5x

# speed_up_video(input_file, output_file, speed_factor)

import argparse
import ffmpeg
import os

# Parse command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--start', help='Start time in HH:MM:SS format')
# parser.add_argument('--end', help='End time in HH:MM:SS format')
# args = parser.parse_args()



# video_name = "Cut Detected Avayar.mp4"
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# output_dir ="Laptop Cut Detected Avayar.mp4"



# (ffmpeg.input(video_name, ss=args.start, to=args.end)
# 	.output(output_dir)
# 	.run())



# def merge_videos(video1, video2, output_file):
#     command = [
#         'ffmpeg',
#         '-i', video1,
#         '-i', video2,
#         '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[outv]',
#         '-map', '[outv]',
#         output_file
#     ]
#     subprocess.run(command)

# # Example usage
# video1 = 'More Cut Detected Avayar.mp4'
# video2 = 'Laptop Cut Detected Avayar.mp4'
# output_file = 'merged_video.mp4'

# merge_videos(video1, video2, output_file)