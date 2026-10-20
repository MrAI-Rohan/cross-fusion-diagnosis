import torch.nn as nn
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class SwinUNet(nn.Module):
    """
    Unet for taking in four input maps.
    Not decoder_channels dynamic cause of the head being hardcoded to upsample 64 channels output.
    Could be dynamic if conv+upsample block  frequency in head was decided based on decoder_channels[-1]. 
    """
    def __init__(self, encoder_channels, num_classes=1):
        super().__init__()

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[256, 128, 64],
            n_blocks=3,
        )

        self.head = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(32, 1, kernel_size=1)
                )

    def forward(self, features):
        x = self.decoder(features)
        x = self.head(x)
        return x