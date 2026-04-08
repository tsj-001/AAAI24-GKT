import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from gkt.model_bank.dynamic_operator import TensorWithMeta, DynamicConv2D, DynamicLinear, \
    DynamicHSigmoid, DynamicReLU, DynamicHSwish, DynamicBatchNorm, DynamicIdentity


def make_divisible(v, divisor=1, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_eight_divisible(v):
    return int(make_divisible(v / 8) * 8)


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
            DynamicLinear(channel, channel // reduction, bias=False),
            DynamicReLU(inplace=True),
            DynamicLinear(channel // reduction, channel, bias=False),
            DynamicHSigmoid()
        )

    def forward(self, x):
        if not isinstance(x, TensorWithMeta):
            indices = range(x.size(1))
        else:
            indices = x.indices
            x = x.input
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = TensorWithMeta(input=y, indices=indices)
        y = self.fc(y)
        indices = y.indices
        y = y.input
        y = y.view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return TensorWithMeta(input=out, indices=indices)

class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False,
                 nl='RE', conv_layer=DynamicConv2D, norm_layer=DynamicBatchNorm):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            nlin_layer = DynamicReLU
        elif nl == 'HS':
            nlin_layer = DynamicHSwish
        else:
            raise NotImplementedError
        if se:
            se_layer = SEModule
        else:
            se_layer = DynamicIdentity

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
        out = self.conv(x)
        if not self.use_res_connect:
            return out
        else:
            return out + x


class MobileNetV3DyDepthWidth(nn.Module):
    def __init__(self, num_classes,
                 dropout_rate=0.1,
                 ordered_width=True,
                 has_contrastive_head=False,):
        super(MobileNetV3DyDepthWidth, self).__init__()
        self.ordered_width = ordered_width
        self.has_contrastive_head = has_contrastive_head
        # k, expansion, ic, oc, se, none-linear, stride
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
        self.depth = [2, 3, 4, 2, 3]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs = HSwish(inplace=True)

        self.stage_list = nn.ModuleList()
        for _i in range(0, 6):
            self.stage_list.append(self._make_mobile_stage(MobileBottleneck, stage_setting[_i]))

        # self.conv2 = nn.Conv2d(160, 960, 1, 1, 0, bias=False)
        self.conv2 = DynamicConv2D(160, 960, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(960, 1280, 1, 1, 0)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()



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
        x = x.input
        x = self.bn2(x)
        x = self.hs(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        x = self.hs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def update_indices(self):
        chosen_width = self.proxy[1]
        out_channel_indices_dict = {}
        for i in range(1, 6):
            stage_i = self.stage_list[i]
            chosen_width_i = chosen_width[i-1]
            for m_name, m in stage_i.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    if chosen_width_i >= 1.0 or self.ordered_width:
                        m.indices_channel = torch.tensor(range(m.weight.size(0)))
                    else:
                        chosen_channel_num_i = make_eight_divisible(chosen_width_i * m.weight.size(0))
                        output_name_key = 'stage{}_{}_{}'.format(i, m.weight.size(0), chosen_channel_num_i)
                        if output_name_key in out_channel_indices_dict.keys():
                            m.indices_channel = out_channel_indices_dict[output_name_key]
                        else:
                            indices_channel = torch.randperm(m.weight.size(0))[:chosen_channel_num_i].sort()[0]
                            m.indices_channel = indices_channel
                            out_channel_indices_dict[output_name_key] = indices_channel

    def set_running_depth(self, proxy):
        self.proxy = copy.deepcopy(proxy)
        self.depth = self.proxy[0]
        for _i in range(1, 6):
            self.stage_list[_i].apply(lambda m: setattr(m, 'width', self.proxy[1][_i-1]))
        self.update_indices()


if __name__ == '__main__':
    torch.manual_seed(0)
    model = MobileNetV3DyDepthWidth(1000, 0.0, ordered_width=False)
    model.set_running_depth([[1, 2, 3, 1, 1], [0.5, 0.6, 0.3, 0.7, 0.9]])
    input_test = torch.randn(2, 3, 224, 224)
    output = model(input_test)
    print(output.shape)
