import torch
import torch.nn as nn
import copy
import numpy as np
from gkt.model_bank.dynamic_operator import DynamicConv2D, DynamicLinear, DynamicBatchNorm, DynamicReLU, TensorWithMeta

def make_divisible(v, divisor=1, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



class WideBasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = DynamicBatchNorm(in_planes)
        self.conv1 = DynamicConv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = DynamicBatchNorm(out_planes)
        self.conv2 = DynamicConv2D(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = DynamicReLU(inplace=True)

        if in_planes != out_planes:
            self.shortcut = DynamicConv2D(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, ste_depth=False):
        out = self.bn1(x)
        out = self.relu(out)
        skip_x = x if isinstance(self.shortcut, nn.Identity) else out
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        indices = out.indices
        out = out.input
        out = self.dropout(out)
        out = TensorWithMeta(input=out, indices=indices)
        out = self.conv2(out)
        if isinstance(self.shortcut, nn.Identity):
            out_skip = skip_x.input if isinstance(skip_x, TensorWithMeta) else skip_x
            indices = out.indices
            out = out.input
            if ste_depth:
                out = out - out.detach() + self.shortcut(out_skip)
            else:
                out += self.shortcut(out_skip)
        else:
            out_skip = self.shortcut(skip_x)
            indices = out.indices
            out = out.input
            if ste_depth:
                out = out - out.detach() + out_skip.input
            else:
                out += out_skip.input
        out = TensorWithMeta(input=out, indices=indices)
        return out


class WideResNetDyDepthWidth(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, init_width_mul=1.0,
                 ordered_width=True, ordered_depth=True, has_contrastive_head=False, ste_depth=False):
        super(WideResNetDyDepthWidth, self).__init__()

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)

        # n_stages_org = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n_stages = [16,
                    int(make_divisible(16 * widen_factor * init_width_mul / 8) * 8),
                    int(make_divisible(32 * widen_factor * init_width_mul / 8) * 8),
                    int(make_divisible(64 * widen_factor * init_width_mul / 8) * 8)]
        self.n_stages = n_stages

        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_wide_stage(WideBasicBlock, n_stages[0], n_stages[1], n, dropout_rate, stride=1)
        self.stage2 = self._make_wide_stage(WideBasicBlock, n_stages[1], n_stages[2], n, dropout_rate, stride=2)
        self.stage3 = self._make_wide_stage(WideBasicBlock, n_stages[2], n_stages[3], n, dropout_rate, stride=2)
        self.bn1 = DynamicBatchNorm(n_stages[3])
        self.relu = DynamicReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = DynamicLinear(n_stages[3], num_classes)
        self.init_width_mul = init_width_mul
        self.ordered_width = ordered_width
        self.ordered_depth = ordered_depth
        self.ste_depth = ste_depth
        for i in range(3):
            getattr(self, 'stage{}'.format(i + 1)).apply(lambda m: setattr(m, 'init_width_mul', init_width_mul))
            getattr(self, 'stage{}'.format(i + 1)).apply(lambda m: setattr(m, 'stage_id', i+1))
        self._init_params()

        proxy = [[n, n, n], [1.0, 1.0, 1.0]]
        self.set_running_depth(proxy)
        self.depth = self.proxy[0]
        self.has_contrastive_head = has_contrastive_head
        self.output_feature = False
        if self.has_contrastive_head:
            self.contrastive_head = nn.Sequential(
                DynamicLinear(n_stages[3], 128),
                DynamicReLU(),
                DynamicLinear(128, 128)
            )

    @staticmethod
    def _make_wide_stage(block, in_planes, out_planes, num_blocks, dropout_rate, stride):
        stride_list = [stride] + [1] * (int(num_blocks) - 1)
        in_planes_list = [in_planes] + [out_planes] * (int(num_blocks) - 1)
        blocks = []

        for _in_planes, _stride in zip(in_planes_list, stride_list):
            blocks.append(block(_in_planes, out_planes, dropout_rate, _stride))

        return nn.Sequential(*blocks)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        for stage_id in range(3):
            stage_i = getattr(self, 'stage{}'.format(stage_id + 1))
            block_num = len(stage_i)
            if not self.ordered_depth:
                chosen_block_id = [0] + sorted(np.random.choice(range(block_num-1), self.depth[stage_id]-1, replace=False)+1)
            else:
                chosen_block_id = range(self.depth[stage_id])
            for block_id, block in enumerate(stage_i):
                if block_id in chosen_block_id:
                    out = block(out)
                else:
                    if self.ste_depth:
                        out = block(out, ste_depth=self.ste_depth)
        out = self.bn1(out)
        out = self.relu(out)
        indices = out.indices
        out = out.input
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        feature = out if not self.has_contrastive_head else self.contrastive_head(TensorWithMeta(out, indices)).input
        out = TensorWithMeta(input=out, indices=indices)
        out = self.linear(out)
        out = out.input

        return (out, feature) if self.output_feature else out

    def update_indices(self):
        chosen_width = self.proxy[1]
        for i in range(3):
            chosen_width_i = chosen_width[i]
            chosen_channel_num_i = int(
                make_divisible(self.n_stages[i + 1] / self.init_width_mul * chosen_width_i / 8) * 8)
            original_channel_num = int(make_divisible(self.n_stages[i + 1] / self.init_width_mul / 8) * 8)
            if chosen_width_i >= 1.0 or self.ordered_width:
                indices_i = torch.tensor(range(chosen_channel_num_i))
            else:
                indices_i = torch.tensor(sorted(np.random.choice(range(original_channel_num), chosen_channel_num_i, replace=False)))
            stage_i = getattr(self, 'stage{}'.format(i + 1))
            stage_i.apply(lambda m: setattr(m, 'indices_channel', indices_i))

    def set_running_depth(self, proxy):
        self.proxy = copy.deepcopy(proxy)
        self.depth = self.proxy[0]
        self.stage1.apply(lambda m: setattr(m, 'width', self.proxy[1][0]))
        self.stage2.apply(lambda m: setattr(m, 'width', self.proxy[1][1]))
        self.stage3.apply(lambda m: setattr(m, 'width', self.proxy[1][2]))
        self.update_indices()




if __name__ == '__main__':
    torch.manual_seed(0)
    net = WideResNetDyDepthWidth(28, 10, 0.3, 10, init_width_mul=2.0, ordered_width=False, ordered_depth=False, has_contrastive_head=True, ste_depth=True)
    net.output_feature = True
    proxy = [[2, 3, 1], [0.2, 0.7, 0.8]]
    net.set_running_depth(proxy)
    tmp_in = torch.randn(2, 3, 32, 32)
    tmp_out = net(tmp_in)
    print(tmp_out.shape)
