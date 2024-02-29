from data_preparation.actions import Action
import cv2
import torch
import numpy as np
from model.slowfast_model import SlowFast  # Ensure this import matches your project structure
from torchvision import transforms
from data_preparation.config import CFG  # Ensure this import matches your project structure
from data_preparation.PackPathwayTransform import PackPathway
from data_preparation.util import get_transformer  # Ensure this import matches your project structure

# Assuming device setup as before
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your trained SlowFast model
model = SlowFast.load_from_checkpoint("checkpoints/last-v1.ckpt")
model.eval()
model.to(device)

# Prepare video capture
cap = cv2.VideoCapture('archery.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))

transform = get_transformer('test')

actions_per_second = []
pack_pathway = PackPathway()

frame_stack = []

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    frame = transform(image=frame)['image']
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess each frame and append it to the frame stack
    processed_frame = preprocess_frame(frame)  # Preprocess the frame
    frame_stack.append(processed_frame)  # Append the processed frame

    # When the number of frames in the stack equals the fps, process them
    if len(frame_stack) == CFG.num_frames:
        # Stack the frames along a new dimension
        frame_stack = torch.stack(frame_stack)
        frame_stack = torch.permute(frame_stack, (1, 0, 2, 3))
        
        # Apply PackPathway or any additional preprocessing if required
        frame_stack = pack_pathway(frame_stack)

        # Predict actions for this second
        with torch.no_grad():
            inputs = [i.to(device)[None, ...] for i in frame_stack]
            outputs = model(inputs)
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(outputs)
            pred_classes = preds.topk(k=3).indices[0]
            pred_class_names = [Action().action[int(i)] for i in pred_classes]
            actions_this_second = pred_class_names

        # Log or use the actions_this_second as needed
        print(f"Actions this second: {actions_this_second}")
        actions_per_second.append(actions_this_second)

        # Clear the frame stack for the next second
        frame_stack = []

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Output actions per second if needed
print(f"Actions per second: {actions_per_second}")
