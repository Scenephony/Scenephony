import torch
from torch import nn, Tensor
from dataclasses import dataclass
from modules.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig

@dataclass
class ScenephonyModelConfig:
    """Dataclass to hold the configuration of the Scenephony model.
    
    Attributes:
        hidden_frame_h: Height of the hidden frame.
        hidden_frame_w: Width of the hidden frame.
    """
    sample_frames = 15
    hidden_frame_h = 224
    hidden_frame_w = 224


class ScenephonyModel(nn.Module):
    """The main Scenephony model."""

    def __init__(self, config: ScenephonyModelConfig) -> None:
        super(ScenephonyModel, self).__init__()
        self.config = config
	self.feature_extractor = CNNFeatureExtractor(CNNFeatureExtractorConfig(
            pretrained=True,
            num_classes=self.config.num_output_classes
        ))

    def forward(self, x: Tensor) -> Tensor:
        """Transforms input video frames to audio notes.
        
        Args:
            x: Input video frames of shape (N, T, C, H, W).
               Where N is the batch size, T is the number of frames, C is the number of channels, 
               H is the height and W is the width.
        """
        pass
