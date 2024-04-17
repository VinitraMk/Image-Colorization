import torch.nn as nn
import torch

class ConvNet(nn.Module):

    def __block(in_c, out_c, pad = 0):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_c, out_channels = out_c),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_c, out_channels = out_c, padding = pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def __init__(self):
        super().__init__()

        self.block1 = self.__block(1, 8)
        self.block2 = self.__block(8, 16)
        self.block3 = self.__block(16, 32)
        self.block4 = self.__block(32, 64)

    def forward(self, inp):

        x = self.block1(inp)
        print('x1 shape', x.shape)
        x = self.block2(x)
        print('x2 shape', x.shape)
        x = self.block3(x)
        print('x3 shape', x.shape)
        x = self.block4(x)
        print('x4 shape', x.shape)
        return x



