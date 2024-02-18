import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from torchvision import transforms
from torch.utils.data import Dataset


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


class ActionDataset(Dataset):
    def __init__(self):
        self.all_videos = self.get_actions_video()
        self.pose = self.get_pose()

    def get_actions_video(self):
        video_paths = []

        data_path = 'data_preparation/actions'
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name) # Get the path to the class

            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    file_path = os.path.join(class_path, file)
                    video_paths.append((file_path, class_name))
        return video_paths

    def get_pose(self):
        pose = []

        for i in range(len(self.all_videos)):
            cap = cv2.VideoCapture(self.all_videos[i][0]) # Get the video path
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():

                    # Read feed
                    ret, frame = cap.read()
                    
                    # Check if frame is not received properly then break the loop
                    if not ret:
                        print("Failed to grab frame")
                        break

                    # Process the frame with MediaPipe Holistic.
                    # Convert the BGR image to RGB.
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Process the image and draw landmarks.
                    results = holistic.process(image)
                    # Convert the RGB image back to BGR.
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Visualization logic comes here (if needed)

                    # Show to screen
                    cv2.imshow('OpenCV Feed', frame)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()

        return pose
                
    def __len__(self):
        return len(self.all_videos)
    
    



# cap = cv2.VideoCapture(0)
# # Set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():

#         # Read feed
#         ret, frame = cap.read()

#         # Show to screen
#         cv2.imshow('OpenCV Feed', frame)

#         # Break gracefully
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
        

ActionDataset()