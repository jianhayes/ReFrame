""" ReFrame: A Resource-Friendly Cloud-Assisted On-Device Deep Learning Framework for Vision Services
Modified from ResNet in PyTorch.
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: xiejianhang@bjtu.edu.cn; jianhang.xie@my.cityu.edu.hk
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.non_learnable_module import NLModule, GroupConv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nlconv_outchans=-1,
                 nlcv='group', nlf=('max', 'avg'), groups=-1, shuffle=True,
                 seed_filters=None, option='kaiming_normal'):
        super(BasicBlock, self).__init__()
        self.seed_filters = seed_filters
        self.nlconv_outchans = nlconv_outchans
        self.left = nn.Sequential(
            NLModule(in_planes, planes, stride=stride, nlconv_outchans=self.nlconv_outchans,
                     nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                     seed_filter=self.seed_filters[0], option=option),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            NLModule(planes, planes, stride=1, nlconv_outchans=self.nlconv_outchans,
                     nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                     seed_filter=self.seed_filters[1], option=option),
            nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def get_layer(self, name):
        return getattr(self, name)

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, nlconv_outchans=-1,
                 nlcv='groups', nlf=('max', 'avg'), groups=-1, shuffle=True,
                 seed_filters=None, option='kaiming_normal'):
        super(Bottleneck, self).__init__()
        self.seed_filters = seed_filters
        self.nlconv_outchans = nlconv_outchans
        self.left = nn.Sequential(
            NLModule(in_planes, planes, stride=1, nlconv_outchans=self.nlconv_outchans,
                     nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                     seed_filter=self.seed_filters[0], option=option),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            NLModule(planes, planes, stride=stride, nlconv_outchans=self.nlconv_outchans,
                     nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                     seed_filter=self.seed_filters[1], option=option),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            NLModule(planes, self.expansion * planes, stride=1, nlconv_outchans=self.nlconv_outchans,
                     nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                     seed_filter=self.seed_filters[2], option=option),
            nn.BatchNorm2d(self.expansion * planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NLResNet(nn.Module):
    def __init__(self, block, num_blocks, init_weights=True, num_classes=1000, imgsz=224,
                 nlconv_outchans=-1, nlcv='group', nlf=('max', 'avg'), groups=-1, shuffle=True,
                 seed_filters=None, init_option='kaiming_normal'):
        super(NLResNet, self).__init__()
        assert isinstance(seed_filters, list) or isinstance(seed_filters, type(None))
        if block is BasicBlock:
            repeat = 2
        elif block is Bottleneck:
            repeat = 3
        else:
            raise NotImplementedError(f'block {block} not implemented.')

        if isinstance(seed_filters, list) and len(seed_filters) != repeat * sum(num_blocks):
            raise NotImplementedError(f'num of seed filters is incorrect.')

        self.seed_filters = [-1 for _ in range(repeat * sum(num_blocks))] if seed_filters is None else seed_filters
        sf = torch.IntTensor(self.seed_filters)
        self.register_buffer("register_sf", sf)

        self.in_planes = 64
        self.imgsz = imgsz
        self.nlconv_outchans = nlconv_outchans
        split1 = repeat*num_blocks[0]
        split2 = repeat*sum(num_blocks[:2])
        split3 = repeat*sum(num_blocks[:3])

        if self.imgsz == 224:
            # input_size is (224, 224)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.imgsz == 32:
            # input_size is (32, 32)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            raise NotImplementedError(f'Image size {imgsz} not implemented.')

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, nlconv_outchans=nlconv_outchans,
                                       nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                                       seed_filters=self.seed_filters[:split1], option=init_option, repeat=repeat)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, nlconv_outchans=nlconv_outchans,
                                       nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                                       seed_filters=self.seed_filters[split1:split2], option=init_option, repeat=repeat)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, nlconv_outchans=nlconv_outchans,
                                       nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                                       seed_filters=self.seed_filters[split2:split3], option=init_option, repeat=repeat)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, nlconv_outchans=nlconv_outchans,
                                       nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                                       seed_filters=self.seed_filters[split3:], option=init_option, repeat=repeat)

        self.classifier = nn.Sequential(nn.Linear(512*block.expansion, num_classes))

        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, nlconv_outchans=-1,
                    nlcv='group', nlf=('max', 'avg'), groups=-1, shuffle=True,
                    seed_filters=None, option='kaiming_normal', repeat=2):
        strides = [stride] + [1]*(num_blocks-1)
        # split seed filters in repeat step
        seed_filters = [seed_filters[i:i+repeat] for i in range(0, len(seed_filters), repeat)]
        layers = []
        idx = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, nlconv_outchans,
                                nlcv=nlcv, nlf=nlf, groups=groups, shuffle=shuffle,
                                seed_filters=seed_filters[idx], option=option))
            self.in_planes = planes * block.expansion
            idx = idx + 1
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # learnable (trainable) parameters initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def print_nlconv_weights(self):
        for m in self.modules():
            if isinstance(m, GroupConv):
                print(m.weight)

    def print_nlconv_size(self):
        for m in self.modules():
            if isinstance(m, GroupConv):
                print(m.weight.size())

    def print_seed_filters(self):
        print(self.register_sf)

    def get_layer(self, name):
        return getattr(self, name)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.imgsz == 224:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def NLGroupResNet18(nlconv_outchans=-1, num_classes=10, imgsz=224, init_weights=True,
                    nlcv='group', nlf=('max', 'mix_avg'), groups=-1, shuffle=True,
                    seed_filters=None, init_option='kaiming_normal'):
    return NLResNet(BasicBlock, [2, 2, 2, 2], init_weights=init_weights, num_classes=num_classes,
                    imgsz=imgsz, nlcv=nlcv, nlf=nlf, nlconv_outchans=nlconv_outchans, groups=groups, shuffle=shuffle,
                    seed_filters=seed_filters, init_option=init_option)


def NLGroupResNet34(nlconv_outchans=-1, num_classes=10, imgsz=224, init_weights=True,
                    nlcv='group', nlf=('max', 'mix_avg'), groups=-1, shuffle=True,
                    seed_filters=None, init_option='kaiming_normal'):
    return NLResNet(BasicBlock, [3, 4, 6, 3], init_weights=init_weights, num_classes=num_classes,
                    imgsz=imgsz, nlcv=nlcv, nlf=nlf, nlconv_outchans=nlconv_outchans, groups=groups, shuffle=shuffle,
                    seed_filters=seed_filters, init_option=init_option)


def NLGroupResNet50(nlconv_outchans=-1, num_classes=10, imgsz=224, init_weights=True,
                    nlcv='group', nlf=('max', 'mix_avg'), groups=-1, shuffle=True,
                    seed_filters=None, init_option='kaiming_normal'):
    return NLResNet(Bottleneck, [3, 4, 6, 3], init_weights=init_weights, num_classes=num_classes,
                    imgsz=imgsz, nlcv=nlcv, nlf=nlf, nlconv_outchans=nlconv_outchans, groups=groups, shuffle=shuffle,
                    seed_filters=seed_filters, init_option=init_option)


def NLGroupResNet101(nlconv_outchans=-1, num_classes=10, imgsz=224, init_weights=True,
                     nlcv='group', nlf=('max', 'mix_avg'), groups=-1, shuffle=True,
                     seed_filters=None, init_option='kaiming_normal'):
    return NLResNet(Bottleneck, [3, 4, 23, 3], init_weights=init_weights, num_classes=num_classes,
                    imgsz=imgsz, nlcv=nlcv, nlf=nlf, nlconv_outchans=nlconv_outchans, groups=groups, shuffle=shuffle,
                    seed_filters=seed_filters, init_option=init_option)


def NLGroupResNet152(nlconv_outchans=-1, num_classes=10, imgsz=224, init_weights=True,
                     nlcv='group', nlf=('max', 'mix_avg'), groups=-1, shuffle=True,
                     seed_filters=None, init_option='kaiming_normal'):
    return NLResNet(Bottleneck, [3, 8, 36, 3], init_weights=init_weights, num_classes=num_classes,
                    imgsz=imgsz, nlcv=nlcv, nlf=nlf, nlconv_outchans=nlconv_outchans, groups=groups, shuffle=shuffle,
                    seed_filters=seed_filters, init_option=init_option)


if __name__ == '__main__':
    from torchprofile import profile_macs

    nlcv = 'group'
    nlf = ('max', 'mix_avg')

    data = torch.rand(1, 3, 32, 32)
    model = NLGroupResNet18(imgsz=32, init_weights=False, nlconv_outchans=768,
                            nlcv=nlcv, nlf=nlf, groups=4, shuffle=True, seed_filters=None)
    # model = NLGroupResNet50(32)
    print(model)
    x = model(data)

    flops = profile_macs(model, data) / 1e6
    print("mac is {}".format(flops))

    print(x.shape)

    from torchinfo import summary

    summary(model, (1, 3, 32, 32), depth=10)

