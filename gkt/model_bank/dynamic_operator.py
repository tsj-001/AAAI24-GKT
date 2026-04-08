import torch
import torch.nn as nn
import torch.nn.functional as F


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


# tensor with meta
class TensorWithMeta():
    def __init__(self, input, indices):
        self.input = input
        self.indices = indices
    def __add__(self, other):
        if isinstance(other, TensorWithMeta):
            return TensorWithMeta(input=self.input + other.input, indices=self.indices)
        elif isinstance(other, torch.Tensor):
            return TensorWithMeta(input=self.input + other, indices=self.indices)
        else:
            raise NotImplementedError


class DynamicConv2D(nn.Conv2d):
    def forward(self, input):
        weight, bias = self.weight, self.bias
        if self.groups == 1:
            output_indices = getattr(self, 'indices_channel', torch.tensor(range(self.weight.size(0))))
            if isinstance(input, TensorWithMeta):
                input_indices = input.indices
                weight = weight[output_indices][:, input_indices]
                input = input.input
            else:
                weight = weight[output_indices][:, :input.size(1)]
            bias = bias[output_indices] if bias is not None else None
            result = self._conv_forward(input, weight, bias)
            return TensorWithMeta(input=result, indices=output_indices)
        elif self.groups == self.weight.size(0):
            if isinstance(input, TensorWithMeta):
                input_indices = input.indices
                weight = weight[input_indices]
                input = input.input
            else:
                weight = weight[:input.size(1)]
                input_indices = range(input.size(1))
            bias = bias[input_indices] if bias is not None else None
            result = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, input.size(1))
            return TensorWithMeta(input=result, indices=input_indices)
        else:
            raise NotImplementedError


class DynamicLinear(nn.Linear):
    def forward(self, input):
        weight, bias = self.weight, self.bias
        output_indices = getattr(self, 'indices_channel', range(self.weight.size(0)))
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            weight = weight[output_indices][:, input_indices]
            input = input.input
        else:
            weight = weight[output_indices][:, :input.size(-1)]

        bias = bias[output_indices] if bias is not None else None
        result = F.linear(input, weight, bias)
        return TensorWithMeta(input=result, indices=output_indices)


class DynamicBatchNorm(nn.BatchNorm2d):
    def forward(self, input):
        weight = self.weight
        bias = self.bias
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            input = input.input
        else:
            channel_dim = 1 if len(input.size()) == 4 else -1
            c = input.size(channel_dim)
            input_indices = range(c)

        y = nn.functional.batch_norm(
            input,
            self.running_mean[input_indices],
            self.running_var[input_indices],
            weight[input_indices],
            bias[input_indices],
            self.training,
            self.momentum,
            self.eps)
        return TensorWithMeta(input=y, indices=input_indices)

class DynamicLayerNorm(nn.LayerNorm):
    def forward(self, input):
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            input = input.input
        else:
            channel_dim = 1 if len(input.size()) == 4 else -1
            c = input.size(channel_dim)
            input_indices = range(c)
        weight, bias = self.weight, self.bias
        if bias is not None:
            bias = bias[input_indices]
        weight = weight[input_indices]
        # out = super().forward(input)
        out = F.layer_norm(input, (len(input_indices),), weight, bias, self.eps)
        return TensorWithMeta(input=out, indices=input_indices)


class DynamicReLU(nn.ReLU):
    def forward(self, input):
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            input = input.input
        else:
            channel_dim = 1 if len(input.size()) == 4 else -1
            c = input.size(channel_dim)
            input_indices = range(c)
        result = F.relu(input, inplace=self.inplace)
        return TensorWithMeta(input=result, indices=input_indices)


class DynamicGeLU(nn.GELU):
    def forward(self, input):
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            input = input.input
        else:
            channel_dim = 1 if len(input.size()) == 4 else -1
            c = input.size(channel_dim)
            input_indices = range(c)
        result = super().forward(input)
        return TensorWithMeta(input=result, indices=input_indices)


class DynamicDropout(nn.Dropout):
    def forward(self, input):
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            input = input.input
        else:
            channel_dim = 1 if len(input.size()) == 4 else -1
            c = input.size(channel_dim)
            input_indices = range(c)
        result = super().forward(input)
        return TensorWithMeta(input=result, indices=input_indices)


class DynamicHSwish(nn.Module):
    def __init__(self, inplace=True):
        super(DynamicHSwish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            input = input.input
        else:
            c = input.size(1)
            input_indices = range(c)
        out = input * F.relu6(input + 3., inplace=self.inplace) / 6.
        return TensorWithMeta(input=out, indices=input_indices)


class DynamicHSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(DynamicHSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            input = input.input
        else:
            c = input.size(1)
            input_indices = range(c)
        out = F.relu6(input + 3., inplace=self.inplace) / 6.
        return TensorWithMeta(input=out, indices=input_indices)

class DynamicIdentity(nn.Identity):
    def forward(self, input):
        if isinstance(input, TensorWithMeta):
            input_indices = input.indices
            input = input.input
        else:
            channel_dim = 1 if len(input.size()) == 4 else -1
            c = input.size(channel_dim)
            input_indices = range(c)
        out = super().forward(input)
        return TensorWithMeta(input=out, indices=input_indices)






