import torch.nn as nn
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder


class SwinUPerNet(nn.Module):
    """UPerNet for four feature maps.
        The decoder_channels and head based on it are """
    def __init__(self, encoder_channels, num_classes=1):
        super().__init__()

        """ SMP UPerNet discards first two inputs so two dummies are given 
            at start so all swin outputs are used."""

        self.decoder = UPerNetDecoder(
            encoder_channels=[1, 1, *encoder_channels],
            decoder_channels=128,
        )

        self.head = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(64, 1, kernel_size=1)
                )


    def forward(self, features):
        dummy = features[0].new_zeros(1, 1, 1, 1)
        x = self.decoder([dummy, dummy]+list(features))
        x = self.head(x)
        return x
