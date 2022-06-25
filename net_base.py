import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, dilation=1):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=dilation, dilation=dilation)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class ERDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(ERDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

        self.calayer = CALayer(in_channels)
        # self.palayer=PALayer(in_channels)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = self.calayer(out)
        # out = self.palayer(out)

        out = out + x

        return out


class CAG(nn.Module):
    def __init__(self, in_channels, layers=3, num_dense_layer=4, growth_rate=16):
        super(CAG, self).__init__()

        self.rdb_module = nn.ModuleDict()
        self.in_channels = in_channels
        self.layers = layers
        self.num_dense_layer = num_dense_layer
        self.growth_rate = growth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((2, layers, in_channels))),requires_grad=True)
        for i in range(layers):
            self.rdb_module.update({'ERDB_0_{}'.format(i): ERDB(in_channels, num_dense_layer, growth_rate)})
            self.rdb_module.update({'ERDB_1_{}'.format(i): ERDB(in_channels, num_dense_layer, growth_rate)})
        self.tail = ERDB(in_channels, num_dense_layer, growth_rate)

    def forward(self, x):

        x_index = [[0 for _ in range(self.layers)] for _ in range(2)]

        x_index[0][0] = self.rdb_module['ERDB_0_{}'.format(0)](x)
        x_index[1][0] = self.rdb_module['ERDB_1_{}'.format(0)](x)

        self.rdb_module['ERDB_1_{}'.format(0)](x)

        for i in range(1, self.layers):
            x_index[0][i] = self.rdb_module['ERDB_0_{}'.format(i)](x_index[0][i-1]) \
                + self.coefficient[0, i-1, :][None, :, None, None] * x_index[1][i-1]
            x_index[1][i] = self.rdb_module['ERDB_1_{}'.format(i)](x_index[1][i-1]) \
                + self.coefficient[1, i - 1, :][None, :, None, None] * x_index[0][i-1]
            # print(i)

        y = self.coefficient[0, -1, :][None, :, None, None] * x_index[0][self.layers-1] \
            + self.coefficient[1, -1, :][None, :, None, None] * x_index[1][self.layers-1]

        y = self.tail(y)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out
