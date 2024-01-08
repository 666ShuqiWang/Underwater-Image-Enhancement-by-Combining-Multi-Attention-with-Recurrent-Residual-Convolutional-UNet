import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d,LeakyReLU,BatchNorm2d, ConvTranspose2d,ReLU
import cv2,datetime,os
from utils import img2tensor,tensor2img
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def encoder_layer(in_channels, out_channels, kernel_size=4, stride=2,
                  padding=1):  # NOTE: Padding here is different from the 'vaild' in tensorflow version of original github
    layer = nn.Sequential(
        Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        BatchNorm2d(out_channels),
        LeakyReLU(0.2)
    )
    return layer


def decoder_layer(in_channels, out_channels, last_layer=False, kernel_size=4, stride=2, padding=1):
    if not last_layer:
        layer = nn.Sequential(
            ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.ReLU()
        )
    else:
        layer = nn.Sequential(
            ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )
    return layer


def discrimiter_layer(in_channels, out_channels, kernel_size=4, stride=2, padding=1, wgan=False):
    if wgan:
        layer = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(out_channels),
            LeakyReLU(0.2)
        )
    else:
        layer = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            LeakyReLU(0.2)
        )
    return layer
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=1):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat((avgout, maxout), dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t#时间步time-step
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out
class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out
class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class GeneratorNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(GeneratorNet, self).__init__()
        n1 = 64
        t=2
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)  # 64
        self.Conv2 = conv_block(ch_in=64, ch_out=128)  # 64 128
        self.Conv3 = conv_block(ch_in=128, ch_out=256)  # 128 256
        self.Conv4 = conv_block(ch_in=256, ch_out=512)  # 256 512
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)  # 512 1024

        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)

        #self.Up5 = up_conv(ch_in=1024, ch_out=512)  # 1024 512
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        #self.Up4 = up_conv(ch_in=512, ch_out=256)  # 512 256
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        #self.Up3 = up_conv(ch_in=256, ch_out=128)  # 256 128
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        #self.Up2 = up_conv(ch_in=128, ch_out=64)  # 128 64
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)  # 64
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        print('Conv1',x1.shape)
        q1 = self.cbam1(x1)
        print('q1',q1.shape)
        x1 = self.cbam1(x1) + x1
        print('cbam1(x1) + x1',x1.shape)

        x2 = self.Maxpool(x1)
        print('Maxpool',x2.shape)
        x2 = self.Conv2(x2)
        print('Conv2',x2.shape)
        x2 = self.cbam2(x2) + x2
        print('cbam2(x2) + x2',x2.shape)

        x3 = self.Maxpool(x2)
        print('Maxpool',x3.shape)
        x3 = self.Conv3(x3)
        print('Conv3',x3.shape)
        x3 = self.cbam3(x3) + x3
        print('cbam3(x3) + x3',x3.shape)

        x4 = self.Maxpool(x3)
        print('Maxpool',x4.shape)
        x4 = self.Conv4(x4)
        print('Conv4',x4.shape)
        x4 = self.cbam4(x4) + x4
        print('cbam4(x4) + x4',x4.shape)

        x5 = self.Maxpool(x4)
        print('Maxpool',x5.shape)
        x5 = self.Conv5(x5)
        print('Conv5',x5.shape)

        # decoding + concat path
        d5 = self.Up5(x5)
        print('Up5',d5.shape)
        x4 = self.Att5(g=d5, x=x4)#r2u把这一行注释掉就好了
        print('Att5',x4.shape)
        d5 = torch.cat((x4, d5), dim=1)
        print('cat',d5.shape)
        d5 = self.Up_RRCNN5(d5)
        print('Up_RRCNN5',d5.shape)

        d4 = self.Up4(d5)
        print('Up4',d4.shape)
        x3 = self.Att4(g=d4, x=x3)
        print('Att4',x3.shape)
        d4 = torch.cat((x3, d4), dim=1)
        print('cat',d4.shape)
        d4 = self.Up_RRCNN4(d4)
        print('Up_RRCNN4',d4.shape)

        d3 = self.Up3(d4)
        print('Up3',d3.shape)
        x2 = self.Att3(g=d3, x=x2)
        print('Att3',x2.shape)
        d3 = torch.cat((x2, d3), dim=1)
        print('cat',d3.shape)
        d3 = self.Up_RRCNN3(d3)
        print('Up_RRCNN3',d3.shape)

        d2 = self.Up2(d3)
        print('Up2',d2.shape)
        x1 = self.Att2(g=d2, x=x1)
        print('Att2',x1 .shape)
        d2 = torch.cat((x1, d2), dim=1)
        print('cat',d2.shape)
        d2 = self.Up_RRCNN2(d2)
        print('Up_RRCNN2',d2.shape)

        d1 = self.Conv(d2)
        print('Conv',d1.shape)
        d1 = torch.tanh(d1)#！！！！！！！1
        print('tanh',d1.shape)
        return d1



if __name__ == "__main__":
    netG = GeneratorNet().to(device)
    input_x = cv2.imread('3.jpg')

    input_x = torch.tensor(input_x).float().to(device)
    input_x = input_x.permute(2, 0, 1)
    #print(input_x.shape)
    input_x = torch.unsqueeze(input_x, 0)
    #print(input_x.shape)
    output = GeneratorNet()
    output_img = netG(input_x)
    #print(output_img.shape)
    # output_img = torch.squeeze(output_img, 0)
    # output_img = output_img.permute(1, 2, 0)
    # output_img = np.asanyarray(output_img.data.cpu())
    output_img = tensor2img(output_img.data.cpu())
#    print(output_img.size())
    cv2.imwrite('output_img.jpg', output_img)

