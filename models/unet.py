import torch.nn as nn
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class SwinUNet(nn.Module):
    """
    UNet for four Swin feature maps (e1..e4, strides 4/8/16/32).

    smp's UnetDecoder treats the first input as the same-resolution map and
    discards it, so a placeholder is placed in front. Skips are then e3, e2, e1
    and all four maps are used.

    decoder_channels is fixed because the head is hardcoded to upsample 64
    channels. It could be made dynamic by deriving the conv+upsample block
    frequency in the head from decoder_channels[-1].
    """
    def __init__(self, encoder_channels, num_classes=1):
        super().__init__()

        assert len(encoder_channels) == 4, "SwinUNet expects exactly four feature maps"

        self.decoder = UnetDecoder(
            encoder_channels=[1, *encoder_channels],  # leading placeholder, discarded by smp
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
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, features):
        assert len(features) == 4, f"expected 4 feature maps, got {len(features)}"
        dummy = features[0].new_zeros(1, 1, 1, 1)  # same dtype/device as the features
        x = self.decoder([dummy, *features])
        return self.head(x)