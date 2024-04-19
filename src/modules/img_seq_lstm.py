from dataclasses import dataclass
from torch import nn, Tensor
import torch


@dataclass
class ImgSeqLSTMConfig:
    """Configuration class for ImgSeqLSTM.
    
    Attributes:
        input_size: Number of features in the input.
        hidden_size: Number of features in the hidden state h.
        num_layers: Number of recurrent layers.
        output_size: Number of features in the output.
    """
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int


class ImgSeqLSTM(nn.Module):
    """LSTM model for image sequence to audio embeddings."""

    def __init__(self, config: ImgSeqLSTMConfig):
        super(ImgSeqLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(config.hidden_size, config.output_size * 2)
        self.fc2 = nn.Linear(config.output_size * 2, config.output_size)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Converts a sequence of image features to audio embeddings.
        
        The input sequence of image features should be ordered sequence of extracted features from
        frame samples of a video. The output is the audio embeddings for the entire video, which is
        time-agnostic.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
        
        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
