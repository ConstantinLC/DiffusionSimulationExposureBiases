import torch.nn as nn


class DilatedResNet(nn.Module):
    def __init__(self, condChannels: int, dataChannels: int, blocks: int = 4, features: int = 48, dilate: bool = True, **kwargs):
        super().__init__()
        self.encoderConv = nn.Conv2d(condChannels, features, kernel_size=3, stride=1, dilation=1, padding=1)

        self.blocks = nn.ModuleList([])
        for _ in range(blocks):
            dils = [2, 4, 8, 4, 2] if dilate else [1, 1, 1, 1, 1]
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features, kernel_size=3, stride=1, dilation=dils[0], padding=dils[0]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features, kernel_size=3, stride=1, dilation=dils[1], padding=dils[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features, kernel_size=3, stride=1, dilation=dils[2], padding=dils[2]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features, kernel_size=3, stride=1, dilation=dils[3], padding=dils[3]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features, kernel_size=3, stride=1, dilation=dils[4], padding=dils[4]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        self.decoderConv = nn.Conv2d(features, dataChannels, kernel_size=3, stride=1, dilation=1, padding=1)

    def forward(self, x, time=None):
        x = self.encoderConv(x)
        skip = x
        for block in self.blocks:
            x = block(x) + skip
            skip = x
        return self.decoderConv(x)
