import torch
import numpy as np


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.in3 = torch.nn.InstanceNorm2d(32, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(32)
        self.res2 = ResidualBlock(32)
        # self.res3 = ResidualBlock(32)
        # self.res4 = ResidualBlock(32)
        # self.res5 = ResidualBlock(32)
        # Upsampling Layers
        self.deconv1 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.in4 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv2 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        # print("X", X.shape)
        y = self.relu(self.in1(self.conv1(X)))
        # print(y.shape)
        y = self.relu(self.in2(self.conv2(y)))
        # print(y.shape)
        y = self.relu(self.in3(self.conv3(y)))
        # print(y.shape)
        y = self.res1(y)
        y = self.res2(y)
        # y = self.res3(y)
        # y = self.res4(y)
        # y = self.res5(y)
        # print(y.shape)
        y = self.relu(self.in4(self.deconv1(y)))
        # print(y.shape)
        y = self.relu(self.in5(self.deconv2(y)))
        # print(y.shape)
        y = self.deconv3(y)
        # print(y.shape)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, onnx=False):
        super(ConvLayer, self).__init__()
        self.onnx = onnx
        val = np.int16(kernel_size // 2)
        reflection_padding = kernel_size // 2  # (val, val, val, val)
        print("reflection_padding", reflection_padding, type(reflection_padding))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        # self.reflection_pad.padding = np.array(
        #     self.reflection_pad.padding, dtype=np.int32
        # )
        print("conv", in_channels, out_channels)
        if self.onnx:
            self.conv2d = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=reflection_padding,
            )
        else:
            self.conv2d = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,  # padding=reflection_padding
            )

    def forward(self, x):
        if self.onnx:
            out = x
        else:
            out = self.reflection_pad(x)

        out2 = self.conv2d(out)
        return out2


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        print("conv", channels, channels)

        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = np.int16(kernel_size // 2)
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        print(
            "convt",
            in_channels,
            out_channels,
            kernel_size,
            self.upsample,
            kernel_size // 2,
        )
        self.conv2d = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size + 1,
            self.upsample,
            # padding=1
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x_in = x
        # if self.upsample:
        #     x_in = torch.nn.functional.interpolate(
        #         x_in, mode="nearest", scale_factor=self.upsample
        #     )
        # out = self.reflection_pad(x_in)
        out = x_in
        out = self.conv2d(out)
        return out
