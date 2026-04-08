from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileBottleneckStoDepth(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False,
                 nl='RE', conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileBottleneckStoDepth, self).__init__()

        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = HSwish
        else:
            raise NotImplementedError

        if se:
            se_layer = SEModule
        else:
            se_layer = nn.Identity

        self.conv = nn.Sequential(OrderedDict([
            # pw
            ('conv1', conv_layer(inp, exp, 1, 1, 0, bias=False)),
            ('bn1', norm_layer(exp)),
            ('act1', nlin_layer(inplace=True)),
            # dw
            ('conv2', conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False)),
            ('bn2', norm_layer(exp)),
            ('act2', nlin_layer(inplace=True)),
            ('se', se_layer(exp)),
            # pw-linear
            ('conv3', conv_layer(exp, oup, 1, 1, 0, bias=False)),
            ('bn3', norm_layer(oup)),
        ]))

        self.prob = kwargs['cur_prob']
        self.sampler = torch.distributions.bernoulli.Bernoulli(self.prob)

        self.mul_enable = kwargs.get('mul_enable', True)

    def forward(self, x):
        skip_flag = True if (self.training and self.sampler.sample() == 0) else False
        mul_flag = True if (not self.training and self.mul_enable) else False

        if skip_flag:
            return x

        out = self.conv(x)

        if mul_flag:
            out = out * self.prob

        if self.use_res_connect:
            return x + out
        else:
            return out


class MobileNetV3StoDepth(nn.Module):
    def __init__(self, num_classes, dropout_rate, **kwargs):  # prob_0_l, mul_enable
        super(MobileNetV3StoDepth, self).__init__()

        # k, exp, ic, oc, se, nl, s
        num_blocks = [1, 2, 3, 4, 2, 3]
        stage_setting = [
            [
                [3, 16, 16, 16, False, 'RE', 1],
            ], [
                [3, 64, 16, 24, False, 'RE', 2],
                [3, 72, 24, 24, False, 'RE', 1],
            ], [
                [5, 72, 24, 40, True, 'RE', 2],
                [5, 120, 40, 40, True, 'RE', 1],
                [5, 120, 40, 40, True, 'RE', 1],
            ], [
                [3, 240, 40, 80, False, 'HS', 2],
                [3, 200, 80, 80, False, 'HS', 1],
                [3, 184, 80, 80, False, 'HS', 1],
                [3, 184, 80, 80, False, 'HS', 1],
            ], [
                [3, 480, 80, 112, True, 'HS', 1],
                [3, 672, 112, 112, True, 'HS', 1],
            ], [
                [5, 672, 112, 160, True, 'HS', 2],
                [5, 960, 160, 160, True, 'HS', 1],
                [5, 960, 160, 160, True, 'HS', 1],
            ]
        ]

        prob_0_l = kwargs.get('prob_0_l', [1, 0.9])
        assert isinstance(prob_0_l, list) and len(prob_0_l) == 2, \
            "The range of probability should be a list of two elements " \
            "or a predefined string (linear_decay or uniform)."
        prob_0_l = np.insert(np.linspace(*prob_0_l, 9), (0, 0, 1, 3, 6, 7), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        prob_stage_list = iter(torch.split(torch.from_numpy(prob_0_l), num_blocks))

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs = HSwish(inplace=True)

        self.stage_list = nn.ModuleList()
        for _i in range(0, 6):
            kwargs['cur_prob_list'] = next(prob_stage_list)
            self.stage_list.append(self._make_mobile_stage(MobileBottleneckStoDepth, stage_setting[_i], **kwargs))

        self.conv2 = nn.Conv2d(160, 960, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(960, 1280, 1, 1, 0)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

    @staticmethod
    def _make_mobile_stage(block, stage_setting, **kwargs):
        blocks = []
        for (k, exp, ic, oc, se, nl, s), prob in zip(stage_setting, kwargs['cur_prob_list']):
            kwargs['cur_prob'] = prob
            blocks.append(block(ic, oc, k, s, exp, se, nl, **kwargs))
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hs(x)

        for stage in self.stage_list:
            x = stage(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.hs(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        x = self.hs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    net = MobileNetV3StoDepth(1000, 0.0, prob_0_l=[0.1, 0.1])
    tmp_out = None
    for _i in range(1000):
        net.train() if _i % 2 == 0 else net.eval()
        tmp_in = torch.randn(2, 3, 32, 32)
        tmp_out = net(tmp_in)
    print(tmp_out.shape)
