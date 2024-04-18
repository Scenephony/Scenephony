from dataclasses import dataclass
from torch import nn, Tensor
import torchvision.models as models
import torch

@dataclass
class CNNFeatureExtractorConfig:
    """Configuration class for CNNFeatureExtractor.
    
    Attributes:
        pretrained: Whether to use the pretrained model weights.
        num_classes: Number of features in the output layer.
    """
    pretrained: bool = True
    num_classes: int

class CNNFeatureExtractor(nn.Module):
    """Model to extract features from a sequence of images using ResNet50 and process through FC layers."""

    def __init__(self, config: CNNFeatureExtractorConfig):
        super(CNNFeatureExtractor, self).__init__()
        # Load the pretrained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=config.pretrained)
        # Replace the output layer with an Identity to use the features directly
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()  # Bypass the final FC layer

        # Additional layers to process the features
        self.fc1 = nn.Linear(num_features, config.num_classes * 2)
        self.fc2 = nn.Linear(config.num_classes * 2, config.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Extracts features from a sequence of input images and processes through FC layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 3, H, W)
        
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        features = self.resnet50(x)

        features = features.view(batch_size, seq_len, -1)
        features = features.mean(dim=1)

        # Fully connected layers to refine the features
        features = self.fc1(features)
        features = self.relu(features)
        output = self.fc2(features)
        return output