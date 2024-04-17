import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# Constants
W = 256  # Width of each frame
H = 256  # Height of each frame

# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=True)

# Keep only the feature extraction part
model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

class VideoFrameDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.videos = [d for d in self.folder_path.iterdir() if d.is_dir()]
        self.transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.CenterCrop((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_path = self.videos[index]
        frames = list(video_path.glob('*.jpg'))
        images = [self.transform(Image.open(frame).convert('RGB')) for frame in frames]
        return torch.stack(images)

def extract_features(folder_path, batch_size=1):
    dataset = VideoFrameDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    video_features = []
    
    with torch.no_grad():
        for video_frames in dataloader:
            batch_features = []
            for frames in video_frames:
                features = model(frames)
                features = features.view(features.size(0), -1)  # Flatten the features
                batch_features.append(features)
            video_features.append(torch.stack(batch_features))

    return video_features

folder_path = 'path_to_key_frames_folder'
features_per_video = extract_features(folder_path)
for video_features in features_per_video:
    print(video_features.shape)  # Each shape will be [N, Feature_length], wh
