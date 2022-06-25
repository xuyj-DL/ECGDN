"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numpy as np


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()
        self.n_feat = n_feat
        self.scale_unetfeats = scale_unetfeats

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level4 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level5 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level6 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.encoder_level4 = nn.Sequential(*self.encoder_level4)
        self.encoder_level5 = nn.Sequential(*self.encoder_level5)
        self.encoder_level6 = nn.Sequential(*self.encoder_level6)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        self.down45 = DownSample(n_feat, scale_unetfeats)
        self.down56 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        self.coefficient = nn.Parameter(
            torch.Tensor(np.ones((2, 2, n_feat + scale_unetfeats))), requires_grad=True)

        # Cross Stage Feature Fusion (CSFF)
        # if csff:
        #     self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        #     self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        #     self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
        #                                bias=bias)

    def forward(self, x1, x2, encoder_outs=None):
        ##### cross level 1

        enc1 = self.encoder_level1(x1)
        enc4 = self.encoder_level4(x2)

        x1 = enc1 + self.coefficient[0, 0, :self.n_feat][None, :, None, None] * enc4
        x1 = self.down12(x1)

        x2 = enc4 + self.coefficient[1, 0, :self.n_feat][None, :, None, None] * enc1
        x2 = self.down45(x2)

        enc2 = self.encoder_level2(x1)
        enc5 = self.encoder_level5(x2)

        x1 = enc2 + self.coefficient[0, 1, :self.n_feat + self.scale_unetfeats][None, :, None, None] * enc5
        x1 = self.down23(x1)

        x2 = enc5 + self.coefficient[1, 1, :self.n_feat + self.scale_unetfeats][None, :, None, None] * enc2
        x2 = self.down56(x2)

        enc3 = self.encoder_level3(x1)
        enc6 = self.encoder_level6(x2)

        return [enc1, enc2, enc3, enc4, enc5, enc6]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False):
        super(Decoder, self).__init__()

        self.n_feat = n_feat
        self.scale_unetfeats = scale_unetfeats

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level4 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level5 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level6 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.decoder_level4 = nn.Sequential(*self.decoder_level4)
        self.decoder_level5 = nn.Sequential(*self.decoder_level5)
        self.decoder_level6 = nn.Sequential(*self.decoder_level6)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.skip_attn4 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn5 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

        self.up54 = SkipUpSample(n_feat, scale_unetfeats)
        self.up65 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

        self.coefficient = nn.Parameter(
            torch.Tensor(np.ones((2, 2, n_feat + scale_unetfeats * 2))), requires_grad=True)

    def forward(self, outs, decoder_outs=None):
        enc1, enc2, enc3, enc4, enc5, enc6 = outs

        dec3 = self.decoder_level3(enc3)
        dec6 = self.decoder_level6(enc6)

        x1 = dec3 + self.coefficient[0, 0, :self.n_feat + self.scale_unetfeats * 2][None, :, None, None] * dec6
        x1 = self.up32(x1, self.skip_attn2(enc2))

        x2 = dec6 + self.coefficient[1, 0, :self.n_feat + self.scale_unetfeats * 2][None, :, None, None] * dec3
        x2 = self.up65(x2, self.skip_attn2(enc5))

        dec2 = self.decoder_level2(x1)
        dec5 = self.decoder_level5(x2)

        x1 = dec2 + self.coefficient[0, 1, :self.n_feat + self.scale_unetfeats][None, :, None, None] * dec5
        x2 = dec5 + self.coefficient[1, 1, :self.n_feat + self.scale_unetfeats][None, :, None, None] * dec2

        x1 = self.up21(x1, self.skip_attn1(enc1))
        x2 = self.up54(x2, self.skip_attn4(enc4))

        dec1 = self.decoder_level1(x1)
        dec4 = self.decoder_level4(x2)

        return [dec1, dec2, dec3, dec4, dec5, dec6]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)

        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = x + y
        return x


##########################################################################
class UNetCross(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCross, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, img1, img2):
        x1 = self.conv1(img1)
        x2 = self.conv2(img2)

        encs = self.encoder(x1, x2)

        decs = self.decoder(encs)

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3])

        return x1, x2


from SRHAN import Layer_Attention_Module_S


class UNetCrossLA(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLA, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = Layer_Attention_Module_S(2, n_feat)

    def forward(self, img1, img2):
        x1 = self.conv1(img1)
        x2 = self.conv2(img2)

        encs = self.encoder(x1, x2)

        decs = self.decoder(encs)

        la_feats = list()

        la_feats.append(decs[0])
        la_feats.append(decs[3])

        # res = self.gp(x)
        feature_LA = self.LA(torch.stack(la_feats[:], dim=1))  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3] + feature_LA + x2)

        return x1, x2


class UNetCrossLA2(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLA2, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = PLA(n_feat)

    def forward(self, img1, img2):
        x1 = self.conv1(img1)
        x2 = self.conv2(img2)

        encs = self.encoder(x1, x2)

        decs = self.decoder(encs)

        la_feats = list()

        la_feats.append(decs[0])
        la_feats.append(decs[3])

        # res = self.gp(x)
        feature_LA = self.LA(torch.stack(la_feats[:], dim=1))  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3] + feature_LA + x2)

        return x1, x2


from ffa import Group


class UNetCrossLAColor(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = Layer_Attention_Module_S(3, n_feat)

        # Color Extrator
        self.color_conv_in = conv(2, n_feat, 1, bias=bias)
        self.color_extrator = Group(conv, n_feat, kernel_size, blocks=3)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)

        encs = self.encoder(x1, x2)

        decs = self.decoder(encs)

        color_x = self.color_conv_in(color_x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)

        la_feats = list()
        la_feats.append(decs[0])
        la_feats.append(decs[3])
        la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(torch.stack(la_feats[:], dim=1))  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3] + feature_LA + x2)

        return x1, x2, color_out


from ffa import GroupHS


class UNetCrossLAColor2(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor2, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = Layer_Attention_Module_S(3, n_feat)

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        color_x = self.color_conv_in(color_x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)

        la_feats = list()
        la_feats.append(decs[0])
        la_feats.append(decs[3])
        la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(torch.stack(la_feats[:], dim=1))  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3] + feature_LA + x2)

        return x1, x2, color_out


class UNetCrossLAColor3(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor2, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = Layer_Attention_Module_S(3, n_feat)

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        self.coefficient = nn.Parameter(
            torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        color_x = self.color_conv_in(color_x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)

        la_feats = list()
        la_feats.append(decs[0] * self.coefficient[0, :][None, :, None, None])
        la_feats.append(decs[3] * self.coefficient[1, :][None, :, None, None])
        la_feats.append(color_feats * self.coefficient[2, :][None, :, None, None])

        # res = self.gp(x)
        feature_LA = self.LA(torch.stack(la_feats[:], dim=1))  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3] + feature_LA + x2)

        return x1, x2, color_out


from net_base import ERDB


class UNetCrossLAColor4(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor4, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = Layer_Attention_Module_S(3, n_feat)

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        self.coefficient = nn.Parameter(
            torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = ERDB(16, 4, 16)
        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        color_x = self.color_conv_in(color_x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)

        la_feats = list()
        la_feats.append(decs[0] * self.coefficient[0, :][None, :, None, None])
        la_feats.append(decs[3] * self.coefficient[1, :][None, :, None, None])
        la_feats.append(color_feats * self.coefficient[2, :][None, :, None, None])

        # res = self.gp(x)
        feature_LA = self.LA(torch.stack(la_feats[:], dim=1))  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3] + feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)
        dehaze = self.feature_fusion(dehaze)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out


class HSNet(nn.Module):
    def __init__(self, n_feat=int(40), kernel_size=3, bias=False):
        super(HSNet, self).__init__()

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=6)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

    def forward(self, x):
        color_x = self.color_conv_in(x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)
        return color_feats, color_out


class HSNet2(nn.Module):
    def __init__(self, n_feat=40, kernel_size=3, bias=False):
        super(HSNet2, self).__init__()

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator1 = GroupHS(conv, n_feat, kernel_size, blocks=6)
        self.color_extrator2 = GroupHS(conv, n_feat, kernel_size, blocks=6)
        self.color_conv_mid = conv(n_feat, n_feat, 1, bias=bias)
        self.color_conv_out = conv(n_feat, 5, 1, bias=bias)
        self.color_conv_out2 = conv(5, 2, 1, bias=bias)

    def forward(self, x):
        color_x = self.color_conv_in(x)
        color_feats1 = self.color_extrator1(color_x)
        color_feats2 = self.color_extrator2(color_feats1)
        color_mid = self.color_conv_mid(color_feats1 + color_feats2)
        color_mid = torch.relu(color_mid)
        color_out = self.color_conv_out(color_mid)
        color_out = torch.relu(color_out)
        color_out = self.color_conv_out2(x + color_out)

        return color_mid, color_out


class PriorFusion(nn.Module):
    def __init__(self, n_feats):
        super(PriorFusion, self).__init__()
        self.conv_f_1 = conv(2, n_feats, 3, bias=False)
        self.conv_f_2 = conv(1, n_feats, 3, bias=False)
        self.conv_f_3 = conv(n_feats, n_feats, 3, bias=False)

        self.c_a = nn.Sequential(*[
            #torch.nn.AdaptiveAvgPool2d(1),
            conv(2, 1, 1, bias=False),
            nn.Sigmoid()
        ])
        self.s_a = nn.Sequential(*[
            #torch.nn.AdaptiveAvgPool2d(1),
            conv(1, 1, 1, bias=False),
            nn.Sigmoid()
        ])

    def forward(self, f, s, c):
        c1 = self.c_a(c)
        s1 = self.s_a(s)

        f1 = self.conv_f_1(c * c1)
        f2 = self.conv_f_2(s * s1)
        f3 = self.conv_f_3(f1 + f2 + f)
        return f3 + f


class FusionModule(nn.Module):
    def __init__(self, n_feats, n_blocks=3):
        super(FusionModule, self).__init__()

        self.blocks = nn.ModuleDict()
        self.pfs = nn.ModuleDict()
        self.n_blocks = n_blocks
        for i in range(n_blocks):
            self.pfs.update({'psf_{}'.format(i): PriorFusion(n_feats)})
            self.blocks.update({'blocks_{}'.format(i): ERDB(n_feats, 4, 16)})

    def forward(self, f_feats, s_feats, c_feats):
        x = f_feats
        for i in range(self.n_blocks):
            x = self.pfs['psf_{}'.format(i)](x, s_feats, c_feats)
            x = self.blocks['blocks_{}'.format(i)](x)
        return x+f_feats


class FusionModule2(nn.Module):
    def __init__(self, n_feats, n_blocks=3):
        super(FusionModule2, self).__init__()

        self.blocks = nn.ModuleDict()
        # self.pfs = nn.ModuleDict()
        self.n_blocks = n_blocks
        for i in range(n_blocks):
            # self.pfs.update({'psf_{}'.format(i): PriorFusion(n_feats)})
            self.blocks.update({'blocks_{}'.format(i): ERDB(n_feats, 4, 16)})

    def forward(self, f_feats, s_feats, c_feats):
        x = f_feats
        for i in range(self.n_blocks):
            # x = self.pfs['psf_{}'.format(i)](x, s_feats, c_feats)
            x = self.blocks['blocks_{}'.format(i)](x)
        return x + f_feats


# ---- LA for ours----#
class PLA(nn.Module):
    def __init__(self, n_feats=40):
        super(PLA, self).__init__()
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.conv2 = conv(n_feats, n_feats, kernel_size=3, bias=False)
        self.p_layer = nn.Sequential(*[
            torch.nn.AdaptiveAvgPool2d(1),
            conv(n_feats, n_feats, kernel_size=1, bias=False),
            nn.ReLU()
        ])
        self.softmax = nn.Softmax()

    def forward(self, f, s, c):
        # p_feats = self.avgpool(s + c)
        # p_feats = self.conv1(p_feats)
        p_feats = self.p_layer(s + c)

        b, c, h, w = p_feats.size()
        feature_group_reshape = p_feats.view(b, c * h * w, 1)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, 1, c * h * w))
        attention_map = self.softmax(attention_map)  # c * h * w  X c * h * w

        attention_feature = torch.bmm(attention_map, feature_group_reshape)  # CHW * 1
        b, chw, _ = attention_feature.size()

        attention_feature = attention_feature.view(b, c, h, w)

        # attention_feature = nn.Softmax(attention_feature)
        f_feats = self.conv2(f) * attention_feature + f
        return f_feats


class PLA2(nn.Module):
    def __init__(self, n_feats=40):
        super(PLA2, self).__init__()
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.convs = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.convc = conv(n_feats, n_feats, kernel_size=1, bias=False)

        self.conv2 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.p_layer = nn.Sequential(*[
            torch.nn.AdaptiveAvgPool2d(1),
            conv(n_feats, n_feats, kernel_size=3, bias=False),
            nn.ReLU()
        ])
        self.softmax = nn.Softmax()

    def forward(self, f, s, c):
        # p_feats = self.avgpool(s + c)
        # p_feats = self.conv1(p_feats)
        s = self.convs(s)
        c = self.convc(c)

        p_feats = self.p_layer(s + c)

        b, c, h, w = p_feats.size()
        feature_group_reshape = p_feats.view(b, c * h * w, 1)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, 1, c * h * w))
        attention_map = self.softmax(attention_map)  # c * h * w  X c * h * w

        attention_feature = torch.bmm(attention_map, feature_group_reshape)  # CHW * 1
        b, chw, _ = attention_feature.size()

        attention_feature = attention_feature.view(b, c, h, w)

        # attention_feature = nn.Softmax(attention_feature)
        f_feats = self.conv2(f) * attention_feature + f
        return f_feats


class PLA3(nn.Module):
    def __init__(self, n, n_feats=40):
        super(PLA3, self).__init__()
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        # self.convs = conv(n_feats, n_feats, kernel_size=1, bias=False)
        # self.convc = conv(n_feats, n_feats, kernel_size=1, bias=False)

        self.conv2 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.conv3 = conv(n_feats * n, n_feats, kernel_size=1, bias=False)

        self.p_layer = nn.Sequential(*[
            torch.nn.AdaptiveAvgPool2d(1),
            conv(n_feats * n, n_feats * n, kernel_size=3, bias=False),
            nn.ReLU()
        ])
        self.softmax = nn.Softmax()

    def forward(self, f, s, c):
        # p_feats = self.avgpool(s + c)
        # p_feats = self.conv1(p_feats)
        # s = self.convs(s)
        # c = self.convc(c)

        p = torch.cat((f, s, c), dim=1)
        p_feats = self.p_layer(p)

        b, c, h, w = p_feats.size()
        feature_group_reshape = p_feats.view(b, c * h * w, 1)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, 1, c * h * w))
        attention_map = self.softmax(attention_map)  # c * h * w  X c * h * w

        attention_feature = torch.bmm(attention_map, feature_group_reshape)  # CHW * 1
        b, chw, _ = attention_feature.size()

        attention_feature = attention_feature.view(b, c, h, w)

        # attention_feature = nn.Softmax(attention_feature)
        f_feats = self.conv2(f) * attention_feature + f
        return f_feats


class PLA4(nn.Module):
    def __init__(self, n_feats=40):
        super(PLA4, self).__init__()
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.conv2 = conv(n_feats, n_feats, kernel_size=3, bias=False)
        self.p_layer = nn.Sequential(*[
            torch.nn.AdaptiveAvgPool2d(1),
            conv(n_feats, n_feats, kernel_size=1, bias=False),
            nn.ReLU()
        ])
        self.softmax = nn.Softmax()

    def forward(self, f, s, c):
        # p_feats = self.avgpool(s + c)
        # p_feats = self.conv1(p_feats)
        # p = torch.cat(f,)
        p_feats = self.p_layer(f + s + c)

        b, c, h, w = p_feats.size()
        feature_group_reshape = p_feats.view(b, c * h * w, 1)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, 1, c * h * w))
        attention_map = self.softmax(attention_map)  # c * h * w  X c * h * w

        attention_feature = torch.bmm(attention_map, feature_group_reshape)  # CHW * 1
        b, chw, _ = attention_feature.size()

        attention_feature = attention_feature.view(b, c, h, w)

        # attention_feature = nn.Softmax(attention_feature)
        f_feats = self.conv2(f) * attention_feature + f
        return f_feats


class PLA5(nn.Module):
    def __init__(self, n, n_feats=40):
        super(PLA5, self).__init__()
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.conv2 = conv(n_feats, n_feats, kernel_size=3, bias=False)
        self.conv3 = conv(n_feats * n, n_feats, kernel_size=3, bias=False)

        self.p_layer = nn.Sequential(*[
            # torch.nn.AdaptiveAvgPool2d(1),
            conv(n_feats * n, n_feats * n, kernel_size=1, bias=False),
            nn.ReLU()
        ])
        self.softmax = nn.Softmax()

    def forward(self, f, s, c):
        # p_feats = self.avgpool(s + c)
        # p_feats = self.conv1(p_feats)
        p = torch.cat((f, s, c), dim=1)
        p_feats = self.p_layer(p)

        b, c, h, w = p_feats.size()
        feature_group_reshape = p_feats.view(b, c, h * w)
        # feature_group_reshape = p_feats
        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, h * w, c))
        attention_map = self.softmax(attention_map)  # c * h * w  X c * h * w

        attention_feature = torch.bmm(attention_map, feature_group_reshape)  # CHW * 1
        b, chw, _ = attention_feature.size()

        attention_feature = attention_feature.view(b, c, h, w)

        attention_feature = self.conv3(attention_feature * p_feats)

        # print(attention_feature.size())
        # attention_feature = nn.Softmax(attention_feature)
        f_feats = self.conv2(f) * attention_feature + f
        return f_feats


class PLA6(nn.Module):
    def __init__(self, n, n_feats=40):
        super(PLA6, self).__init__()
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.conv2 = conv(n_feats * n, n_feats, kernel_size=1, bias=False)
        self.p_layer = nn.Sequential(*[
            torch.nn.AdaptiveAvgPool2d(1),
            conv(n_feats * n, n_feats * n, kernel_size=1, bias=False),
            nn.ReLU()
        ])
        self.softmax = nn.Softmax()

    def forward(self, f, s, c):
        # p_feats = self.avgpool(s + c)
        # p_feats = self.conv1(p_feats)
        p = torch.cat((f, s, c), dim=1)
        p_feats = self.p_layer(p)

        b, c, h, w = p_feats.size()
        feature_group_reshape = p_feats.view(b, c * h * w, 1)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, 1, c * h * w))
        attention_map = self.softmax(attention_map)  # c * h * w  X c * h * w

        attention_feature = torch.bmm(attention_map, feature_group_reshape)  # CHW * 1
        b, chw, _ = attention_feature.size()

        attention_feature = attention_feature.view(b, c, h, w)

        # attention_feature = nn.Softmax(attention_feature)
        f_feats = self.conv2(p * attention_feature) + f
        return f_feats


class Layer_Attention_Module_S(nn.Module):
    def __init__(self, n, c):
        super(Layer_Attention_Module_S, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.scale = nn.Parameter(torch.zeros(1))
        self.n = n
        self.c = c
        # print(n,c)
        self.conv = nn.Conv2d(self.n * self.c, self.c, kernel_size=3, padding=1)

    def forward(self, feature_group):
        b, n, c, h, w = feature_group.size()
        # print(b,n,c,h,w)
        feature_group_reshape = feature_group.view(b, n, c * h * w)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, c * h * w, n))
        attention_map = self.softmax(attention_map)  # N * N

        attention_feature = torch.bmm(attention_map, feature_group_reshape)  # N * CHW
        b, n, chw = attention_feature.size()
        attention_feature = attention_feature.view(b, n, c, h, w)

        attention_feature = self.scale * attention_feature + feature_group
        b, n, c, h, w = attention_feature.size()
        return self.conv(attention_feature.view(b, n * c, h, w))


class PLA7(nn.Module):
    def __init__(self, n, n_feats=40):
        super(PLA7, self).__init__()
        # self.conv1 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        # self.conv2 = conv(n_feats, n_feats, kernel_size=1, bias=False)
        self.conv3 = conv(n_feats * n, n_feats, kernel_size=1, bias=False)

        self.softmax1 = nn.Softmax(dim=2)

        self.scale1 = nn.Parameter(torch.zeros(1))

        self.p_layer = nn.Sequential(*[
            # torch.nn.AdaptiveAvgPool2d(1),
            conv(n_feats * n, n_feats * n, kernel_size=1, bias=False),
            nn.ReLU()
        ])

    def forward(self, f, s, c):
        p = torch.cat((f, s, c), dim=1)
        p_feats = self.p_layer(p)

        b, c, h, w = p_feats.size()

        feature_group_reshape1 = p_feats.view(b, c, h * w)
        # feature_group_reshape2 = feature_group_reshape1.view(b, h * w, c)

        # --- 1 ----#
        attention_map = torch.bmm(feature_group_reshape1, feature_group_reshape1.view(b, h * w, c))
        attention_map = self.softmax1(attention_map)  # C x C

        attention_feature1 = torch.bmm(attention_map, feature_group_reshape1)  # C X h *w
        # print(attention_feature1.size())
        # b, chw, _ = attention_feature.size()

        attention_feature1 = attention_feature1.view(b, c, h, w)
        attention_feature = self.scale1 * attention_feature1 + p

        # --- 2 --- #
        # f_feats = self.conv3(attention_feature) + f
        return self.conv3(attention_feature) + f


class UNetCrossLAColor5(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor5, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = Layer_Attention_Module_S(3, n_feat)
        #
        # self.coefficient = nn.Parameter(
        #     torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = FusionModule(16, 3)

        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)
        # print(type(color_x))

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        # color_feats, color_out = color_x
        color_x = self.color_conv_in(color_x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)

        la_feats = list()
        la_feats.append(decs[0])
        la_feats.append(decs[3])
        la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(torch.stack(la_feats[:], dim=1))  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3] + feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)

        # print(type(color_x))
        # print(type(dehaze))

        # print(type(color_out))
        dehaze = self.feature_fusion(dehaze, x1, color_out)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out


class UNetCrossLAColor6(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor6, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = Layer_Attention_Module_S(3, n_feat)
        #
        # self.coefficient = nn.Parameter(
        #     torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # # Color Extrator
        # self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        # self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        # self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = FusionModule(16, 3)

        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)
        # print(type(color_x))

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        color_feats, color_out = color_x
        # color_x = self.color_conv_in(color_x)
        # color_feats = self.color_extrator(color_x)
        # color_out = self.color_conv_out(color_x + color_feats)

        la_feats = list()
        la_feats.append(decs[0])
        la_feats.append(decs[3])
        la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(torch.stack(la_feats[:], dim=1))  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(decs[3] + feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)

        dehaze = self.feature_fusion(dehaze, x1, color_out)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out


class UNetCrossLAColor7(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor7, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = PLA(n_feat)
        #
        # self.coefficient = nn.Parameter(
        #     torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # # Color Extrator
        # self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        # self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        # self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = FusionModule(16, 3)

        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)
        # print(type(color_x))

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        color_feats, color_out = color_x
        # color_x = self.color_conv_in(color_x)
        # color_feats = self.color_extrator(color_x)
        # color_out = self.color_conv_out(color_x + color_feats)

        # la_feats = list()
        # la_feats.append(decs[0])
        # la_feats.append(decs[3])
        # la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(decs[3], decs[0], color_feats)  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)

        dehaze = self.feature_fusion(dehaze, x1, color_out)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out


class UNetCrossLAColor8(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor8, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = PLA(n_feat)
        #
        # self.coefficient = nn.Parameter(
        #     torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = FusionModule(16, 3)

        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)
        # print(type(color_x))

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        # color_feats, color_out = color_x
        color_x = self.color_conv_in(color_x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)

        # la_feats = list()
        # la_feats.append(decs[0])
        # la_feats.append(decs[3])
        # la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(decs[3], decs[0], color_feats)  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)

        dehaze = self.feature_fusion(dehaze, x1, color_out)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out


class UNetCrossLAColor9(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor9, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = PLA5(3, n_feat)
        #
        # self.coefficient = nn.Parameter(
        #     torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = FusionModule2(16, 3)

        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)
        # print(type(color_x))

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        # color_feats, color_out = color_x
        color_x = self.color_conv_in(color_x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)

        # la_feats = list()
        # la_feats.append(decs[0])
        # la_feats.append(decs[3])
        # la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(decs[3], decs[0], color_feats)  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)

        dehaze = self.feature_fusion(dehaze, x1, color_out)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out


class UNetCrossLAColor10(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor10, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = PLA6(3, n_feat)
        #
        # self.coefficient = nn.Parameter(
        #     torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # Color Extrator
        self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = FusionModule2(16, 3)

        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)
        # print(type(color_x))

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        # color_feats, color_out = color_x
        color_x = self.color_conv_in(color_x)
        color_feats = self.color_extrator(color_x)
        color_out = self.color_conv_out(color_x + color_feats)

        # la_feats = list()
        # la_feats.append(decs[0])
        # la_feats.append(decs[3])
        # la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(decs[3], decs[0], color_feats)  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)

        dehaze = self.feature_fusion(dehaze, x1, color_out)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out

class UNetCrossLAColor11(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor11, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = PLA7(3, n_feat)
        #
        # self.coefficient = nn.Parameter(
        #     torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # Color Extrator
        self.color_extrator = HSNet2()
        # self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        # self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        # self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = FusionModule2(16, 3)

        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)
        # print(type(color_x))

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        # color_feats, color_out = color_x
        # color_x = self.color_conv_in(color_x)
        # color_feats = self.color_extrator(color_x)
        # color_out = self.color_conv_out(color_x + color_feats)
        color_feats, color_out = self.color_extrator(color_x)
        # la_feats = list()
        # la_feats.append(decs[0])
        # la_feats.append(decs[3])
        # la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(decs[3], decs[0], color_feats)  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)

        dehaze = self.feature_fusion(dehaze, x1, color_out)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out

class UNetCrossLAColor12(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=int(40), scale_unetfeats=int(20), scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(UNetCrossLAColor12, self).__init__()

        act = nn.PReLU()
        self.conv1 = conv(2, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(in_c, n_feat, kernel_size, bias=bias)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.tail1 = conv(n_feat, 1, kernel_size, bias=bias)
        self.tail2 = conv(n_feat, out_c, kernel_size, bias=bias)

        # LA
        self.LA = PLA7(3, n_feat)
        #
        # self.coefficient = nn.Parameter(
        #     torch.Tensor(np.ones((3, n_feat))), requires_grad=True)

        # Color Extrator
        self.color_extrator = HSNet2()
        # self.color_conv_in = conv(5, n_feat, 1, bias=bias)
        # self.color_extrator = GroupHS(conv, n_feat, kernel_size, blocks=4)
        # self.color_conv_out = conv(n_feat, 2, 1, bias=bias)

        # 特征融合
        self.conv_ff_in = conv(6, 16, kernel_size, bias=bias)
        self.feature_fusion = FusionModule(16, 3)

        self.conv_ff_out = conv(16, 3, kernel_size, bias=bias)

    def forward(self, edge, haze, color_x):
        x1 = self.conv1(edge)
        x2 = self.conv2(haze)
        # print(type(color_x))

        encs = self.encoder(x1, x2)
        decs = self.decoder(encs)

        # color_feats, color_out = color_x
        # color_x = self.color_conv_in(color_x)
        # color_feats = self.color_extrator(color_x)
        # color_out = self.color_conv_out(color_x + color_feats)
        color_feats, color_out = self.color_extrator(color_x)
        # la_feats = list()
        # la_feats.append(decs[0])
        # la_feats.append(decs[3])
        # la_feats.append(color_feats)

        # res = self.gp(x)
        feature_LA = self.LA(decs[3], decs[0], color_feats)  # b, n * c, h, w

        x1 = self.tail1(decs[0])
        x2 = self.tail2(feature_LA + x2)

        # concat
        x_feats = torch.cat((x2, x1, color_out), dim=1)
        dehaze = self.conv_ff_in(x_feats)

        dehaze = self.feature_fusion(dehaze, x1, color_out)
        dehaze = self.conv_ff_out(dehaze) + haze
        return x1, dehaze, color_out

if __name__ == '__main__':
    # net = UNetCrossLA()
    # print(net)
    # net = UNetCrossLA()
    # print(net)
    from torchsummary import summary

    netG = UNetCrossLAColor12()
    summary(netG, [(2, 256, 256), (3, 256, 256), (5, 256, 256)], batch_size=1, device="cpu")
    # netG = PLA7(3, 40)
    # summary(netG, [(40, 256, 256), (40, 256, 256), (40, 256, 256)], batch_size=1, device="cpu")
