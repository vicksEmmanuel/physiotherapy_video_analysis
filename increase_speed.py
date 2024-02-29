import subprocess

def speed_up_video(input_file, output_file, speed_factor):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-filter_complex',
        f"[0:v]setpts=PTS/{speed_factor}[v];[0:a]atempo={speed_factor}[a]",
        '-map', '[v]',
        '-map', '[a]',
        output_file
    ]
    subprocess.run(command)

# Example usage
input_file = 'RPReplay_Final1709231743.MP4'
output_file = 'WhatsApp Video 2024-02-29 at 19.03.00 _speed_up.mp4'
speed_factor = 1.5  # Increase speed by 1.5x

speed_up_video(input_file, output_file, speed_factor)
