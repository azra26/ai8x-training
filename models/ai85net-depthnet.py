###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
SimpleNet_v1 network with added residual layers for AI85.
Simplified version of the network proposed in [1].

[1] HasanPour, Seyyed Hossein, et al. "Lets keep it simple, using simple architectures to
    outperform deeper and more complex architectures." arXiv preprint arXiv:1608.06037 (2016).
"""
from torch import nn

import ai8x


class AI85DepthNet(nn.Module):
    """
    Residual SimpleNet Depthnet v1 Model
    """
    def __init__(
            self,
            num_classes=None,
            num_channels=3,
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, padding=1,
                                          bias=False, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=False, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(32, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(32, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                 stride=1, padding=1, bias=bias, **kwargs)
        self.conv7 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2d(64, 512, 1, pool_size=2, pool_stride=2,
                                             padding=0, bias=False, **kwargs)
        self.fc1 = ai8x.Linear(512, 1024, bias=bias, wide=True, **kwargs)                                         
    

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)           # 16x128x128
        x = self.conv2(x)           # 32x64x64
        x = self.conv3(x)           # 32x32x32
        x = self.conv4(x)           # 64x16x16
        x = self.conv5(x)           # 64x8x8
        x = self.conv6(x)           # 64x4x4
        x = self.conv7(x)           # 64x2x2
        x = self.conv8(x)           # 512x1x1
        x = x.view(x.size(0), -1)
        x = self.fc1(x)             # 1024x1x1
        return x


def ai85depthnet(pretrained=False, **kwargs):
    """
    Constructs a Residual SimpleNet v1 model.
    """
    assert not pretrained
    return AI85DepthNet(**kwargs)


models = [
    {
        'name': 'ai85depthnet',
        'min_input': 1,
        'dim': 3,
    },
]
