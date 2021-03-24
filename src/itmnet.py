import torch
import torch.nn as nn
import torch.nn.functional as F

from deepy.nn.model import UNet


class ITMNet(UNet):
    """iTM-Net
    """
    class ConvNormReLU(nn.Module):
        """(conv => BN => ReLU)
        """
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super().__init__()
            self.conv = nn.Sequential(
                conv(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=padding,
                     bias=False),
                normalization(out_channels),
                activation(),
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(
            self, in_channels, out_channels, conv, down_conv,
            normalization, down_kernel_size=3, down_padding=1,
            activation=nn.ReLU):
            super().__init__()
            self.conv = nn.Sequential(
                ITMNet.ConvNormReLU(in_channels=in_channels,
                                    out_channels=in_channels,
                                    conv=conv,
                                    normalization=normalization,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    activation=activation),
                down_conv(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=down_kernel_size,
                          donw_padding=down_padding,
                          stride=2, bias=False),
            )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    class gl_cat(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super().__init__()
            self.conv = ITMNet.double_conv(
                in_channels=in_channels,
                out_channels=out_channels,
                conv=conv,
                normalization=normalization,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation)

        def forward(self, x1, x2):
            x = torch.cat([x1, x2], dim=1)
            x = self.conv(x)
            return x

    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 conv=nn.Conv2d, up_conv=nn.ConvTranspose2d, down_conv=nn.Conv2d,
                 normalization=nn.BatchNorm2d, activation=nn.ReLU,
                 final_activation=nn.Identity):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            max_channels=max_channels,
            conv=conv,
            up_conv=up_conv,
            down_conv=down_conv,
            normalization=normalization,
            activation=activation,
            final_activation=final_activation
        )

        # global encoder
        ge_layers = [
            ITMNet.down(
                in_channels=3,
                out_channels=64,
                conv=conv,
                down_conv=down_conv,
                normalization=normalization,
                down_kernel_size=3,
                down_padding=1,
                activation=activation
            )
        ]
        ge_layers.extend([
            ITMNet.down(
                in_channels=64,
                out_channels=64,
                conv=conv,
                down_conv=down_conv,
                normalization=normalization,
                down_kernel_size=3,
                down_padding=1,
                activation=activation
            )
            for i in reversed(range(4))
        ])
        ge_layers.append(
            ITMNet.ConvNormReLU(
                in_channels=64,
                out_channels=64,
                conv=conv,
                normalization=normalization,
                kernel_size=4,
                padding=0,
                activation=activation
            )
        )
        self.g_down = nn.Sequential(*list(ge_layers))
        self.cat = self.gl_cat(512+64, 512)
        
    def forward(self, x):
        skip_connections = []

        # local encoder
        lx = self.inc(x)
        for l in self.down_blocks:
            skip_connections.append(lx)
            lx = l(lx)
        
        # global encoder
        gx = F.interpolate(x, size=(128, 128),
                           mode="bilinear", align_corners=False)
        gx = self.g_down(gx)
        gx = F.interpolate(gx, size=lx.size()[2:],
                           mode="nearest")
        
        # concat feature maps
        x = self.cat(lx, gx)
    
        # decoder
        for l in self.up_blocks:
            x = l(x, skip_connections.pop())
        x = self.outc(x)
        return x
