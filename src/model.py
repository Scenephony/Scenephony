import torch
from torch import nn, Tensor


class ScenephonyModel(nn.Module):
    """The main Scenephony model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.audio_sample_rate = 44100

    
    def forward(self, x: Tensor) -> Tensor:
        pass
