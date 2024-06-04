"""Code adpted from https://github.com/LeeJunHyun/Image_Segmentation"""

import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(UNet, self).__init__()

        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        down_channels = [img_ch, 32, 64, 128, 256, 512]
        self.down_blocks = nn.ModuleList([])
        for in_, out_ in zip(down_channels[:-1], down_channels[1:]):
            self.down_blocks.append(ConvBlock(in_, out_))

        self.up_conv_blocks = nn.ModuleList([])
        self.up_attn_blocks = nn.ModuleList([])
        self.up_samle_blocks = nn.ModuleList([])
        up_channels = down_channels[::-1][:-1]
        for in_, out_ in zip(up_channels[:-1], up_channels[1:]):
            self.up_conv_blocks.append(UpSample(in_, out_))
            self.up_samle_blocks.append(ConvBlock(out_ * 2, out_))

        self.head = nn.Conv2d(down_channels[1], output_ch, kernel_size=1)

    def forward(self, x):
        down_outputs = []
        for idx, down_block in enumerate(self.down_blocks):
            x = down_block(x)
            if idx != len(self.down_blocks) - 1:
                down_outputs.append(x)
                x = self.pool_layer(x)

        for idx, (up_conv_block, up_sample_block) in enumerate(
            zip(self.up_conv_blocks, self.up_samle_blocks)
        ):
            x = up_conv_block(x)
            x = torch.cat((down_outputs.pop(), x), dim=1)
            x = up_sample_block(x)

        x = self.head(x)
        return x


if __name__ == "__main__":
    model = UNet(3, 20)
    x = torch.randn(1, 3, 512, 512)
    print(model(x).shape)
