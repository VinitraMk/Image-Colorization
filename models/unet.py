
import torch.nn as nn
import torch

class Encoder(nn.Module):

    def __block(self, in_c, out_c, mp_size = 2, dp = 0.2):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = 5),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            #nn.MaxPool2d(mp_size),
            #nn.Dropout(dp)
        )

    def __init__(self):
        super().__init__()
        self.block1 = self.__block(1, 8, 2)
        self.block2 = self.__block(8, 16, 2)
        self.block3 = self.__block(16, 32, 2)
        self.block4 = self.__block(32, 64, 2)
        #self.block5 = self.__block(64, 128, 2)

    def forward(self, x):
        #print('inp', x.size())
        h1 = self.block1(x) #32 -> 28
        #print('h1', h1.size())
        h2 = self.block2(h1) #28 -> 24
        #print('h2', h2.size())
        h3 = self.block3(h2) #24 -> 20
        #print('h3', h3.size())
        h4 = self.block4(h3) #20 -> 16
        #print('h4', h4.size())
        #h5 = self.block5(h4)
        #print('h5', h5.size())

        return [x, h1, h2, h3, h4]

class Decoder(nn.Module):

    def __conv_block(self, in_c, out_c, p = 0.1):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            #nn.Dropout(p)
        )

    def __block(self, in_c, out_c, kernel_size, stride, padding, p = 0.2, out = False):
        if not(out):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(out_c),
                nn.ReLU() if not out else nn.Sigmoid(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(out_c),
                nn.ReLU() if not out else nn.Sigmoid(),
                nn.Identity()
            )

    def __init__(self):
        super().__init__()
        #self.block1 = self.__block(128, 64, 5, 1, 0)
        self.block1 = self.__block(64, 32, 5, 1, 0)
        self.block2 = self.__block(32, 16, 5, 1, 0)
        self.block3 = self.__block(16, 8, 5, 1, 0)
        self.block4 = self.__block(8, 2, 5, 1, 0, True)
        #self.block6 = self.__block(4, 2, 5, 1, 0, True)

        self.cb1 = self.__conv_block(64, 32)
        self.cb2 = self.__conv_block(32, 16)
        self.cb3 = self.__conv_block(16, 8)
        #self.cb4 = self.__conv_block(16, 8)
        #self.cb5 = self.__conv_block(8, 4)

    def forward(self, h):
        #print('inp', h[-1].size()) #16 x 16
        x = self.block1(h[-1]) #16 -> 32
        #print('o1', x.size())

        #print('c1', x.size(), h[-2].size())
        x = torch.concat([x, h[-2]], dim=1) #16
        #print('i1', x.size())
        x = self.cb1(x)
        #print('o2', x.size())
        x = self.block2(x) #16 -> 15
        ##print('o3', x.size())

        #print('c2', x.size(), h[-3].size())
        x = torch.concat([x, h[-3]], dim=1) #8+8 = 16
        #print('i2', x.size())
        x = self.cb2(x) #16 -> 8
        #print('o2', x.size())
        x = self.block3(x) #8 -> 4
        #print('o3', x.size())

        #print('c3', x.size(), h[-4].size())
        x = torch.concat([x, h[-4]], dim=1) #4+4 = 8
        #print('i3', x.size())
        x = self.cb3(x) #8 -> 4
        #print('o4', x.size())
        x = self.block4(x) #4 -> 1
        #print('o5', x.size())

        #x = torch.concat([x, h[-5]], dim = 1)
        #print('i4', x.size())
        #x = self.cb4(x)
        #print('o6', x.size())
        #x = self.block5(x)
        #print('o7', x.size())

        ''' 
        x = torch.concat([x, h[-6]], dim = 1)
        #x = self.cb5(x)
        x = self.block6(x)
        print('fo', x.size())
        '''

        return x

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, ip):
        h = self.encoder(ip)
        op = self.decoder(h)
        return op


