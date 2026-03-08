""" ReFrame: A Resource-Friendly Cloud-Assisted On-Device Deep Learning Framework for Vision Services
Non-learnable Module in PyTorch.
NL filters: max-pooling, avg-pooling, mixed-avg-pooling, mixed-max-pooling
NL conv: standard conv, group conv
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: xiejianhang@bjtu.edu.cn; jianhang.xie@my.cityu.edu.hk
"""

from torch import nn as nn
import torch
import torch.nn.functional as F
import math


def seed_xavier_normal_(tensor, gain=1., generator=None):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0., std, generator=generator)


def seed_kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedPool2d(nn.ModuleDict):
    """
    A mix of avg/max pooling operations with different kernel sizes
    --pool_size: pooling size,
    """
    def __init__(self, in_channels, option='avg', kernel_sizes=(3, 5, 7, 9, 11, 13), stride=1):
        super(MixedPool2d, self).__init__()

        self.pool = option
        self.stride = stride

        num_groups = len(kernel_sizes)
        self.splits = _split_channels(in_channels, num_groups)

        if option == 'avg':
            pool = nn.AvgPool2d
        elif option == 'max':
            pool = nn.MaxPool2d
        else:
            raise NotImplementedError("only avg/max pooling is supported")

        for idx, ks in enumerate(kernel_sizes):
            self.add_module(str(idx), pool(ks, stride=stride, padding=ks // 2))

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]

        if 'avg' in self.pool and self.stride < 2:
            return torch.cat(x_out, 1) - x  # from PoolFormer
        else:
            return torch.cat(x_out, 1)


# computing parameters and FLOPs
# class GroupConv(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, stride, groups, kernel_size=3):
#         super().__init__(in_channels, out_channels, kernel_size, stride,
#                          padding=1, groups=groups, bias=False)
#         self.weight.requires_grad = False
#         if self.bias is not None:
#             self.bias.requires_grad = False


class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, kernel_size=3,
                 generator=None, option='kaiming_normal'):
        super(GroupConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # weight.shape is (out_channels, in_channels // groups, kernel_size)
        weight = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size])
        # print(weight.shape)

        if option == 'kaiming_normal':
            # nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
            seed_kaiming_normal_(weight, mode='fan_out', nonlinearity='relu', generator=generator)
        elif option == 'xavier_normal':
            seed_xavier_normal_(weight, generator=generator)
        elif option == 'std_normal':
            weight.normal_(0., 1., generator=generator)
        else:
            raise NotImplementedError("only kaiming_normal/xavier_normal/std_normal init is supported")

        self.register_buffer("weight", weight)
        self.bias = None
        self.stride = stride
        self.padding = 1
        self.dilation = 1
        self.groups = groups

    def get_layer(self, name):
        return getattr(self, name)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class NLlayer(nn.Module):
    """Non-Learnable Operations Layers.

    Args:
        in_channels (int): Input channels.
        stride (int): Stride.
        padding (int): Padding of pooling and Non-Learnable Convolution. Default: 1.
        nlconv_outchans (int): Non-Learnable Convolution output channels. if -1 is Auto output channels. Default: -1.
        nlcv (str): Non-Learnable Convolution operator. Default: 'group'.
        nlf (Tuple[str, str]): Non-Learnable Convolution operator. Default: '('max', 'mix_avg')'.
        seed_filter (int): Non-Learnable Convolution initial generator seed filter. Default: -1.
        option (str): Non-Learnable Convolution initial distribution function. Default: 'kaiming_normal'.
    """
    def __init__(self, in_channels, stride, padding=1, nlconv_outchans=-1,
                 nlcv='group', nlf=('max', 'mix_avg'), seed_filter=-1, option='kaiming_normal'):
        super(NLlayer, self).__init__()
        self.in_channels = in_channels
        self.stride = stride

        self.nlconv_outchans = in_channels if nlconv_outchans == -1 else nlconv_outchans
        self.NLfilter = nn.ModuleList([nn.Sequential() for _ in range(2)])
        self.g = None if seed_filter == -1 else torch.Generator()
        if self.g is not None:
            self.g.manual_seed(seed_filter)

        # NL Conv
        if nlcv == 'std':
            # self.NLconv = nn.Sequential()
            # self.NLconv.add_module('nlconv', GroupConv(in_channels, self.nlconv_outchans, self.stride, groups=1,
            #                                            generator=self.g, option=option))
            self.NLconv = GroupConv(in_channels, self.nlconv_outchans, self.stride, groups=1,
                                    generator=self.g, option=option)
        elif nlcv == 'group':
            self.groups = math.gcd(in_channels, self.nlconv_outchans)
            # self.NLconv = nn.Sequential()
            # self.NLconv.add_module('nlconv', GroupConv(in_channels, self.nlconv_outchans, self.stride, self.groups,
            #                                            generator=self.g, option=option))
            self.NLconv = GroupConv(in_channels, self.nlconv_outchans, self.stride, self.groups,
                                    generator=self.g, option=option)
        else:
            raise NotImplementedError(f'NL Conv {nlcv} not implemented.')

        # NL filter
        for idx in range(2):
            if nlf[idx] == 'max':
                self.NLfilter[idx] = torch.nn.MaxPool2d(3, self.stride, padding)
            elif nlf[idx] == 'avg':
                self.NLfilter[idx] = torch.nn.AvgPool2d(3, self.stride, padding)
            elif nlf[idx] == 'mix_avg':
                self.NLfilter[idx] = MixedPool2d(in_channels, option='avg',
                                                 kernel_sizes=(3, 5, 7, 9, 11), stride=self.stride)
            elif nlf[idx] == 'mix_max':
                self.NLfilter[idx] = MixedPool2d(in_channels, option='max',
                                                 kernel_sizes=(3, 5, 7), stride=self.stride)
            else:
                raise NotImplementedError(f'NL Conv {nlf[idx]} not implemented.')

    def forward(self, x):
        x1 = F.relu(self.NLconv(x))
        x2 = self.NLfilter[0](x)
        x3 = self.NLfilter[1](x)
        return torch.cat([x1, x2, x3], 1)


class NLModule(nn.Module):
    """Non-Learnable Module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride.
        nlconv_outchans (int): Non-Learnable Convolution output channels. if -1 is Auto output channels. Default: -1.
        nlcv (str): Non-Learnable Convolution operator. Default: 'group'.
        nlf (Tuple[str, str]): Non-Learnable Convolution operator. Default: '('max', 'mix_avg')'.
        groups (int): Groups of Pointwise Convolution. if -1 is math.gcd of Pointwise. Default: 4.
        shuffle (bool): Channel Shuffle. Default: True.
        seed_filter (int): seed_filter for Non-Learnable Convolution. Default: -1.
        option (str): Non-Learnable Convolution initial distribution function. Default: 'kaiming_normal'.
    """
    def __init__(self, in_channels, out_channels, stride, nlconv_outchans=-1,
                 nlcv='group', nlf=('max', 'mix_avg'), groups=4, shuffle=True, seed_filter=-1, option='kaiming_normal'):
        super(NLModule, self).__init__()

        self.stride = stride
        self.shuffle = shuffle
        self.nllayer = NLlayer(in_channels, self.stride, nlconv_outchans=nlconv_outchans,
                               nlcv=nlcv, nlf=nlf, seed_filter=seed_filter, option=option)

        self.nlconv_outchans = in_channels if nlconv_outchans == -1 else nlconv_outchans
        # pointwise conv groups
        self.groups = math.gcd(self.nlconv_outchans, out_channels) if groups == -1 else groups

        self.trans_conv = nn.Sequential(
            nn.Conv2d(self.nlconv_outchans + 2 * in_channels, out_channels,
                      padding=0, kernel_size=1, groups=self.groups)
        )

    def forward(self, x):
        x1 = self.nllayer(x)
        output = self.trans_conv(x1)
        if self.groups > 1 and self.shuffle:
            output = self.channel_shuffle(output, self.groups)
        return output

    def get_layer(self, name):
        return getattr(self, name)

    def channel_shuffle(self, x, group):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % group == 0
        group_channels = num_channels // group

        x = x.reshape(batchsize, group_channels, group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x