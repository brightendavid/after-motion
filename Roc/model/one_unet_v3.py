'''v3:第三阶段与原图进行堆叠'''



import sys
sys.path.append('../')
from unet_parts_paralleling import *



class UNetStage(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(UNetStage, self).__init__()
        factor = 2 if bilinear else 1
        _factor = 1 if bilinear else 2
        # print('factor is : ',_factor)
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        stage_x1 = self.up1(x5, x4)
        stage_x2 = self.up2(stage_x1, x3)
        stage_x3 = self.up3(stage_x2, x2)
        stage_x4 = self.up4(stage_x3, x1)
        logits = self.outc(stage_x4)
        return [logits]

class Cat_Unet(nn.Module):
    def __init__(self, n_channels=5, bilinear=False):
        super(Cat_Unet, self).__init__()
        factor = 2 if bilinear else 1
        _factor = 1 if bilinear else 2
        # print('factor is : ',_factor)
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)


    def forward(self,x1,x2,_x):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        diffY = x2.size()[2] - _x.size()[2]
        diffX = x2.size()[3] - _x.size()[3]
        _x = F.pad(_x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1, _x], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        stage_x1 = self.up1(x5, x4)
        stage_x2 = self.up2(stage_x1, x3)
        stage_x3 = self.up3(stage_x2, x2)
        stage_x4 = self.up4(stage_x3, x1)
        logits = self.outc(stage_x4)
        return [logits]
