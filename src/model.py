import torch
from modules.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig
from modules.img_seq_lstm import ImgSeqLSTM, ImgSeqLSTMConfig
from torch import nn, Tensor
from dataclasses import dataclass

# Configuration class
@dataclass
class ScenephonyModelConfig:
    """Dataclass to hold the configuration of the Scenephony model.

    Attributes:
        sample_frames: Number of frames sampled from the video.
        hidden_frame_h: Height of the frame.
        hidden_frame_w: Width of the frame.
        cnn_num_classes: Number of classes (features) for the CNN output layer.
        lstm_input_size: Number of input features for LSTM.
        lstm_hidden_size: Number of features in the hidden state of LSTM.
        lstm_num_layers: Number of layers in LSTM.
        lstm_output_size: Number of output features from LSTM.
    """
    cnn_feature_size: int = 128
    lstm_input_size: int = 128
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_output_size: int = 10
    sample_frames: int = 15
    hidden_frame_h: int = 224
    hidden_frame_w: int = 224

# Main model class
class ScenephonyModel(nn.Module):
    """The main Scenephony model which processes video frames to audio notes."""

    def __init__(self, config: ScenephonyModelConfig) -> None:
        super(ScenephonyModel, self).__init__()
        self.config = config
        # Initialize CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(CNNFeatureExtractorConfig(pretrained=True, feature_size=config.cnn_feature_size))
        # Initialize LSTM
        self.lstm = ImgSeqLSTM(ImgSeqLSTMConfig(
            input_size=config.lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            output_size=config.lstm_output_size
        ))

    def forward(self, x: Tensor) -> Tensor:
        """Transforms input video frames to audio notes.
        
        Args:
            x: Input video frames of shape (N, T, C, H, W).
               Where N is the batch size, T is the number of frames, C is the number of channels, 
               H is the height and W is the width.

        Returns:
            Tensor of shape (N, output_size), representing audio notes.
        """
        # Process through CNN
        cnn_output = self.cnn(x)  # Expected shape (N, T, num_classes)
        # Process through LSTM
        audio_output = self.lstm(cnn_output)
        return audio_output
