import os
import cv2
import csv
import json

save_path = 'frames_lists/video_frames.csv'
data_path = '/xxxxx/'
video_folder = '/Users/victorumesiobi/Desktop/vicks/Me/Pythorch/physiotherapy/videos'

# This function generates the csv data that holds the frames, frames location, and video properties
def generate_frames(save_path, data_path):
    # Create a new CSV file with a clean state
    csv_file = open(save_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['original_video_id', 'video_id', 'frame_id', 'path', 'labels'])
    csv_file.close()

    # Loop through the folders in the video folder
    for folder_name in os.listdir(data_path):
        json_file_path = os.path.join(data_path, folder_name)
        with open(json_file_path) as json_file:
            data = json.load(json_file)
        # Extract the videoId and framesCount from the JSON data
        video_id = data['videoId']
        frames_count = data['framesCount']
        video_name = data['videoName']
        

        csv_file = open(save_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)

        for frame in data['frames']:
            frame_index = frame['index']
            figures = frame['figures']
            
            # Process the figures in the frame
            for figure in figures:
                figure_id = figure['id']
                class_id = figure['classId']
                object_id = figure['objectId']
                description = figure['description']
                geometry_type = figure['geometryType']
                labeler_login = figure['labelerLogin']
                created_at = figure['createdAt']
                updated_at = figure['updatedAt']
                geometry = figure['geometry']
                
                # Process the geometry points
                exterior_points = geometry['points']['exterior']
                interior_points = geometry['points']['interior']
            
            # Write data to csv
            csv_writer.writerow([video_name, video_id, frame_index, json_file_path, ''])
        csv_file.close()


