import torch
import torch.nn as nn

class CRBlock(nn.Module):
    def __init__(self, growth_factor=32):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=growth_factor,
                                  kernel_size=(3, 3),
                                  padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
    

class CFEBlock(nn.Module):
    def __init__(self, high_channels, low_channels, W=0.4, growth_factor=32):
        """
        Both outputs have same number of channels as the inputs.
        """

        super().__init__()

        self.W = W

        self.cr_high = nn.ModuleList([CRBlock(growth_factor) for i in range(4)])
        self.cr_low = nn.ModuleList([CRBlock(growth_factor) for i in range(4)])
        self.upsample_layers = nn.ModuleList([self.upsample(growth_factor) for i in range(5)])
        self.downsample_layers = nn.ModuleList([self.downsample(growth_factor) for i in range(5)])

        self.cr_high.append(CRBlock(high_channels))
        self.cr_low.append(CRBlock(low_channels))

    def forward(self, inputs):
        e_high, e_low = inputs
        map1_high = self.cr_high[0](torch.cat((e_high, self.upsample_layers[0](e_low)), dim=1))
        map1_low = self.cr_low[0](torch.cat((e_low, self.downsample_layers[0](e_high)), dim=1))

        maps_high = [e_high, map1_high]
        maps_low = [e_low, map1_low]

        for i in range(2, 6):
            inp_high = torch.cat(
                    [
                        maps_high[i-1],
                        self.upsample_layers[i-1](maps_low[i-1]),
                        *maps_high[:i-1][::-1]
                    ], dim=1)
            maps_high.append(self.cr_high[i-1](inp_high))

            inp_low = torch.cat(
                    [
                        maps_low[i-1],
                        self.downsample_layers[i-1](maps_high[i-1]),
                        *maps_low[:i-1][::-1]
                    ], dim=1)

            maps_low.append(self.cr_low[i-1](inp_low))

        out_high = self.W * maps_high[-1] + e_high
        out_low = self.W * maps_low[-1] + e_low

        return out_high, out_low

    def upsample(self, out_channels):
        return nn.LazyConvTranspose2d(out_channels=out_channels, kernel_size=3,
                                stride=2, padding=1, output_padding=1)

    def downsample(self, out_channels):
        return nn.LazyConv2d(out_channels=out_channels, kernel_size=3,
                            stride=2, padding=1)


class CFENet(nn.Module):
    def __init__(self, encoder_channels, W=0.4, growth_factor=32):
        super().__init__()

        assert len(encoder_channels) == 4, "CFENet designed for 4 input maps"

        self.cfe_high1 = nn.Sequential(
            CFEBlock(*encoder_channels[:2], W=W, growth_factor=growth_factor),
            CFEBlock(*encoder_channels[:2], W=W, growth_factor=growth_factor),
        )

        self.cfe_low1 = nn.Sequential(
            CFEBlock(*encoder_channels[2:], W=W, growth_factor=growth_factor),
            CFEBlock(*encoder_channels[2:], W=W, growth_factor=growth_factor),
        )

        self.cfe_mid1 = CFEBlock(*encoder_channels[1:3], W=W, growth_factor=growth_factor)

        self.cfe_high2 = CFEBlock(*encoder_channels[:2], W=W, growth_factor=growth_factor)
        self.cfe_low2 = CFEBlock(*encoder_channels[2:], W=W, growth_factor=growth_factor)

        self.cfe_mid2 = CFEBlock(*encoder_channels[1:3], W=W, growth_factor=growth_factor)

        self.cfe_high3 = CFEBlock(*encoder_channels[:2], W=W, growth_factor=growth_factor)
        self.cfe_low3 = CFEBlock(*encoder_channels[2:], W=W, growth_factor=growth_factor)

        self.W = W

    def forward(self, inputs):
        e1, e2, e3, e4 = inputs

        y1_high, y1_low = self.cfe_high1([e1, e2])
        y2_high, y2_low = self.cfe_low1([e3, e4])

        y1_low, y2_high = self.cfe_mid1([y1_low, y2_high])

        y1_high, y1_low = self.cfe_high2([y1_high, y1_low])
        y2_high, y2_low = self.cfe_low2([y2_high, y2_low])

        y1_low, y2_high = self.cfe_mid2([y1_low, y2_high])

        y1_high, y1_low = self.cfe_high3([y1_high, y1_low])
        y2_high, y2_low = self.cfe_low3([y2_high, y2_low])

        y1_high = self.W * y1_high + e1
        y1_low = self.W * y1_low + e2
        y2_high = self.W * y2_high + e3
        y2_low = self.W * y2_low + e4

        return y1_high, y1_low, y2_high, y2_low
