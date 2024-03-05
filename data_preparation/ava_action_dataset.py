class AVADataset(Dataset):
    def __init__(self, video_paths, annotations, transforms=None):
        self.video_paths = video_paths
        self.annotations = annotations  # Dict or path to annotation file
        self.transforms = transforms

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        annotation = self.annotations[video_path]
        video = load_video(video_path)
        frames, boxes, labels = preprocess_video_and_annotations(video, annotation, self.transforms)
        
        # Convert frames and boxes to tensors
        frames_tensor = torch.as_tensor(frames, dtype=torch.float32)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)

        return frames_tensor, boxes_tensor, labels_tensor
