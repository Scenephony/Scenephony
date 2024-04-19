from dataclasses import dataclass
from torch import nn, Tensor
import torchvision.models as models
import torch


@dataclass
class CNNFeatureExtractorConfig:
    """Configuration class for CNNFeatureExtractor.

    Attributes:
        pretrained: Whether to use the pretrained model weights.
        feature_size: Number of features in each frame's output feature vector.
    """
    feature_size: int
    pretrained: bool = True

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
        self.fc1 = nn.Linear(num_features, config.feature_size * 2)
        self.fc2 = nn.Linear(config.feature_size * 2, config.feature_size)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Extracts features from a sequence of input images and processes through FC layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 3, H, W)

        Returns:
            Output tensor of shape (batch_size, seq_len, feature_size).
        """
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(-1, c, h, w)  # Flatten the batch and sequence dimensions for feature extraction
        features = self.resnet50(x)

        features = features.view(batch_size, seq_len, -1)  # Reshape back to include sequence length
        features = self.fc1(features)  # Apply the first FC layer across each sequence element
        features = self.relu(features)
        return features
