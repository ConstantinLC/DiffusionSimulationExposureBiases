import torch.nn as nn
from neuralop.models import FNO as _FNO


class FNO(nn.Module):
    def __init__(self, condChannels: int, dataChannels: int, modes: list, hidden_channels: int = 64, n_layers: int = 4, **kwargs):
        super().__init__()
        self.model = _FNO(
            n_modes=tuple(modes),
            hidden_channels=hidden_channels,
            in_channels=condChannels,
            out_channels=dataChannels,
            n_layers=n_layers,
        )

    def forward(self, x, time=None):
        return self.model(x)
