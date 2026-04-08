import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(x))
        skip_x = x if isinstance(self.shortcut, nn.Identity) else out

        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(skip_x)

        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes):
        super(ResNet, self).__init__()

        n_stages = [16, 16, 32, 64]

        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(BasicBlock, n_stages[0], n_stages[1], num_blocks[0], stride=1)
        self.stage2 = self._make_stage(BasicBlock, n_stages[1], n_stages[2], num_blocks[1], stride=2)
        self.stage3 = self._make_stage(BasicBlock, n_stages[2], n_stages[3], num_blocks[2], stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(n_stages[3], num_classes)

        self._init_params()

    @staticmethod
    def _make_stage(block, in_planes, out_planes, num_blocks, stride):
        stride_list = [stride] + [1] * (int(num_blocks) - 1)
        in_planes_list = [in_planes] + [out_planes] * (int(num_blocks) - 1)
        blocks = []

        for _in_planes, _stride in zip(in_planes_list, stride_list):
            blocks.append(block(_in_planes, out_planes, _stride))

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
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    torch.manual_seed(0)
    net = ResNet([5, 5, 5], 100)
    tmp_in = torch.randn(2, 3, 32, 32)
    tmp_out = net(tmp_in)
    print(tmp_out.shape)
