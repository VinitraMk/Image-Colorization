import torch.nn as nn
import torch

class ConvNet(nn.Module):

    def __block(self, in_c, out_c, pad1 = 0, pad2 = 0):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 3, padding = pad1),
            #nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_c, out_channels = out_c, kernel_size = 3, padding = pad2),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def __upblock(self, in_c, out_c, pad = 0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_c, out_channels = out_c, kernel_size = 7),
            #nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_c, out_channels = out_c, kernel_size = 3),
            nn.ReLU()
        )

    def __init__(self):
        super().__init__()

        self.block1 = self.__block(1, 8, 0, 1)
        self.block2 = self.__block(8, 16, 0, 1)
        self.block3 = self.__block(16, 32, 0, 1)
        self.block4 = self.__block(32, 64, 0, 1)
        self.block5 = self.__block(64, 128, 0, 1)
        self.block6 = self.__block(128, 128, 1, 1)
        self.block7 = self.__block(128, 128, 1, 1)
        self.block8 = self.__upblock(128, 64)
        self.block9 = self.__upblock(64, 32)
        self.block10 = self.__upblock(32, 16)
        self.block11 = self.__block(16, 8)
        self.conv = nn.Conv2d(in_channels = 8, out_channels = 2, kernel_size = 3, padding = 2)

    def forward(self, inp):
        #print('inp shape', inp.shape, inp.min(), inp.max())
        x = self.block1(inp)
        #print('x1 shape', x.shape)
        x = self.block2(x)
        #print('x2 shape', x.shape)
        x = self.block3(x)
        #print('x3 shape', x.shape)
        x = self.block4(x)
        #print('x4 shape', x.shape)
        x = self.block5(x)
        #print('x5 shape', x.shape)
        x = self.block6(x)
        #print('x6 shape', x.shape)
        x = self.block7(x)
        #print('x7 shape', x.shape)
        x = self.block8(x)
        #print('x8 shape', x.shape)
        x = self.block9(x)
        #print('x9 shape', x.shape)
        x = self.block10(x)
        #print('x10 shape', x.shape)
        x = self.block11(x)
        #print('x11 shape', x.shape)
        x = self.conv(x)
        #print('conv shape', x.shape)
        return x

