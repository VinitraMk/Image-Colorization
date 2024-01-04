import torch.nn as nn
import torch

class Encoder(nn.Module):

    def __block(in_c, out_c, mp_size = 2, dp = 0.2):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(mp_size),
            nn.Dropout(dp)
        )

    def __init__(self):
        super().__init__()
        self.block1 = self.__block(1, 4)
        self.block2 = self.__block(4, 8)
        self.block3 = self.__block(8, 16)
        self.block4 = self.__block(16, 32)

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h4)

        return [h1, h2, h3, h4]

class Decoder(nn.Module):

    def __conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def __block(self, in_c, out_c, kernel_size, padding, out = False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU() if not out else nn.Sigmoid(),
            nn.Dropout(0.2) if not out else nn.Identity(),
        )

    def __init__(self):
        super().__init__()
        self.block1 = self.__block(32, 16, 3, 0)
        self.block2 = self.__block(16, 8, 4, 1)
        self.block3 = self.__block(8, 4, 3, 0)
        self.block4 = self.__block(4, 2, 3, 1, True)

        self.cb1 = self.__conv_block(32, 8)
        self.cb2 = self.__conv_block(16, 8)
        self.cb3 = self.__conv_block(8, 4)

    def forward(self, h):
        x = self.block1(h[-1]) #32 -> 16

        x = torch.concat([x, h[-2]], dim=1) #16+16 = 32
        x = self.cb1(x) #32 -> 8
        x = self.block2(x) #16 -> 8

        x = torch.concat([x, h[-3]], dim=1) #8+8 = 16
        x = self.cb2(x) #16 -> 8
        x = self.block3(x) #8 -> 4

        x = torch.concat([x, h[-4]], dim=1) #4+4 = 8
        x = self.cb3(x) #8 -> 4
        x = self.block4(x) #4 -> 1

class UNet(nn.Module):

    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, ip):
        if self.train:
            h = self.encoder(ip)
            op = self.decoder(h)
            return op
        else:
            with torch.no_grad():
                h = self.encoder(ip)
                op = self.decoder(h)
                return op


