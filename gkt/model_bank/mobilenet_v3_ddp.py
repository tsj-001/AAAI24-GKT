from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False,
                 nl='RE', conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d):
        super(MobileBottleneck, self).__init__()
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

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3DyDepth(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super(MobileNetV3DyDepth, self).__init__()

        # k, exp, ic, oc, se, nl, s
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

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs = HSwish(inplace=True)

        self.stage_list = nn.ModuleList()
        for _i in range(0, 6):
            self.stage_list.append(self._make_mobile_stage(MobileBottleneck, stage_setting[_i]))

        self.conv2 = nn.Conv2d(160, 960, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(960, 1280, 1, 1, 0)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

        self.depth = [2, 3, 4, 2, 3]

    @staticmethod
    def _make_mobile_stage(block, stage_setting):
        blocks = []
        for k, exp, ic, oc, se, nl, s in stage_setting:
            blocks.append(block(ic, oc, k, s, exp, se, nl))
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

        for _i, stage in enumerate(self.stage_list):
            if _i == 0:
                x = stage(x)
            else:
                x = stage[:self.depth[_i - 1]](x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.hs(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        x = self.hs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_running_depth(self, depth):
        self.depth = depth


if __name__ == '__main__':
    torch.manual_seed(0)
    model = MobileNetV3DyDepth(1000, 0.0)
    input_test = torch.randn(2, 3, 224, 224)
    output = model(input_test)
    print(output.shape)
