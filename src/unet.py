import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """ U-Net
    """
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(UNet.double_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                activation(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(UNet.inconv, self).__init__()
            self.conv = UNet.double_conv(in_ch, out_ch, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch, down_conv_layer=nn.Conv2d, activation=nn.ReLU):
            super(UNet.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                padding=1, kernel_size=(3, 3),
                                stride=(2, 2), bias=False),
                UNet.double_conv(in_ch, out_ch, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch, up_conv_layer=nn.ConvTranspose2d, activation=nn.ReLU):
            super(UNet.up, self).__init__()
            self.upconv = up_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                        kernel_size=(4, 4), stride=(2, 2),
                                        padding=1, bias=False)
            self.conv = UNet.double_conv(mid_ch, out_ch, activation=activation)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(UNet.outconv, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.ReLU(inplace=True)
                )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    def __init__(self, n_channels, n_classes,
                 up_conv_layer=nn.ConvTranspose2d,
                 down_conv_layer=nn.Conv2d,
                 activation=nn.ReLU):
        super(UNet, self).__init__()
        self.inc = UNet.inconv(n_channels, 64, activation=activation)
        self.down1 = UNet.down(64, 128, activation=activation)
        self.down2 = UNet.down(128, 256, activation=activation)
        self.down3 = UNet.down(256, 512, activation=activation)
        self.down4 = UNet.down(512, 512, activation=activation)
        self.up1 = UNet.up(512, 1024, 256, activation=activation)
        self.up2 = UNet.up(256, 512, 128, activation=activation)
        self.up3 = UNet.up(128, 256, 64, activation=activation)
        self.up4 = UNet.up(64, 128, 64, activation=activation)
        self.outc = UNet.outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class SEUNet(UNet):
    """ U-Net
    """
    class se_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(SEUNet.se_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                activation(),
                SELayer(out_ch, reduction=16),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(SEUNet.inconv, self).__init__()
            self.conv = SEUNet.se_conv(in_ch, out_ch, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch, down_conv_layer=nn.Conv2d, activation=nn.ReLU):
            super(SEUNet.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                padding=1, kernel_size=(3, 3),
                                stride=(2, 2), bias=False),
                SEUNet.se_conv(in_ch, out_ch, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch, up_conv_layer=nn.ConvTranspose2d, activation=nn.ReLU):
            super(SEUNet.up, self).__init__()
            self.upconv = up_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                        kernel_size=(4, 4), stride=(2, 2),
                                        padding=1, bias=False)
            self.conv = SEUNet.se_conv(mid_ch, out_ch, activation=activation)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(SEUNet.outconv, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.ReLU(inplace=True)
                )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    def __init__(self, n_channels, n_classes,
                 up_conv_layer=nn.ConvTranspose2d,
                 down_conv_layer=nn.Conv2d,
                 activation=nn.ReLU):
        super(SEUNet, self).__init__(n_channels, n_classes)
        self.inc = SEUNet.inconv(n_channels, 64, activation=activation)
        self.down1 = SEUNet.down(64, 128, activation=activation)
        self.down2 = SEUNet.down(128, 256, activation=activation)
        self.down3 = SEUNet.down(256, 512, activation=activation)
        self.down4 = SEUNet.down(512, 512, activation=activation)
        self.up1 = SEUNet.up(512, 1024, 256, activation=activation)
        self.up2 = SEUNet.up(256, 512, 128, activation=activation)
        self.up3 = SEUNet.up(128, 256, 64, activation=activation)
        self.up4 = SEUNet.up(64, 128, 64, activation=activation)
        self.outc = SEUNet.outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x