import torch
import torch.nn as nn

from segmentation_models_pytorch.decoders.deeplabv3.decoder import (
    ASPP,
    SeparableConv2d,
)


class SwinDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for four input maps by Swin Transformer.
    This class uses SMP's components and follows mainly DeepLabV3PlusDecoder
     structure from it to work with Swin.

    Parameters:
        aspp_index: index into encoder_channels for the map passed to ASPP
        high_res_index: index into encoder_channels for the higher-res skip map
    """

    def __init__(
        self,
        encoder_channels,
        aspp_index,
        high_res_index,
        out_channels = 256,
        atrous_rates = (3, 6, 9),
        high_level_out_channels = 48,
        aspp_separable = True,
        aspp_dropout = 0.5,
    ):
        super().__init__()

        self.aspp_index = aspp_index
        self.high_res_index = high_res_index

        # Assumes a standard hierarchical Swin backbone:
        # deepest feature at OS=32, skip feature at OS=4.
        scale_factor = 8

        self.aspp = nn.Sequential(
            ASPP(
                encoder_channels[aspp_index],
                out_channels,
                atrous_rates,
                separable=aspp_separable,
                dropout=aspp_dropout,
            ),
            SeparableConv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.up = nn.Upsample(
            mode="bilinear", scale_factor=scale_factor, align_corners=False
        )

        highres_in_channels = encoder_channels[high_res_index]
        self.block1 = nn.Sequential(
            nn.Conv2d(
                highres_in_channels, high_level_out_channels,
                kernel_size=1, bias=False,
            ),
            nn.BatchNorm2d(high_level_out_channels),
            nn.ReLU(),
        )

        # Paper ablates 1 vs 2 vs 3 refinement convs post-concat and
        # finds 2x(3x3, 256ch) optimal — smp's decoder uses only one; this
        # matches the paper instead.
        self.block2 = nn.Sequential(
            SeparableConv2d(
                high_level_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SeparableConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # ASPP out_channels hard-coded to 256 despite the constructor argument.
        self.head = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(128, 1, kernel_size=1)
                )

    def forward(self, features):
        aspp_features = self.aspp(features[self.aspp_index])
        aspp_features = self.up(aspp_features)

        high_res_features = self.block1(features[self.high_res_index])

        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        
        decoder_features = self.block2(concat_features)
        logits = self.head(decoder_features)
        return logits
