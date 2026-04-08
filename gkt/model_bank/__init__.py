from .wide_resnet import WideResNet
from .resnet import ResNet
from .mobilenet_v3 import MobileNetV3
from .wide_resnet_std import WideResNetStoDepth
from .resnet_std import ResNetStoDepth
from .mobilenet_v3_std import MobileNetV3StoDepth
from .mobilenet_v3_ddw import MobileNetV3DyDepthWidth
from .wide_resnet_ddp import WideResNetDyDepth
from .wide_resnet_ddw import WideResNetDyDepthWidth
from .resnet_ddp import ResNetDyDepth
from .big_resnet_ddw import ResNetDyDepthWidth
from .mobilenet_v3_ddp import MobileNetV3DyDepth
from .vit_ddp import ViTDyDepth
from .vit_ddw import ViTDyDepthWidth
from .swin_ddp import SwinTransformerDyDepth
from .pit_ddp import PiTDyDepth
from .t2t_ddp import T2T_ViTDyDepth
from .cait_ddp import CaiTDyDepth
from . import big_resnet
from . import big_resnet_ddp
from . import big_resnet_ddw
from . import big_resnet_std


from torchvision.models import resnet50, resnet34

# wider resnet
def build_wide_resnet_16_8(num_classes=100, dropout=0.0, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    dy_depth_width = kwargs.get('dy_depth_width', False)
    ordered_width = kwargs.get('ordered_width', True)
    has_contrastive_head = kwargs.get('has_contrastive_head', False)
    if sto_depth:
        return WideResNetStoDepth(16, 8, dropout, num_classes, **kwargs)
    elif dy_depth_width:
        return WideResNetDyDepthWidth(16, 8, dropout, num_classes, ordered_width=ordered_width, has_contrastive_head=has_contrastive_head)
    elif dy_depth:
        return WideResNetDyDepth(16, 8, dropout, num_classes)
    else:
        return WideResNet(16, 8, dropout, num_classes)


def build_wide_resnet_28_10(num_classes=100, dropout=0.0, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    dy_depth_width = kwargs.get('dy_depth_width', False)
    init_width_mul = kwargs.get('init_width_mul', 1.0)
    ordered_width = kwargs.get('ordered_width', True)
    ordered_depth = kwargs.get('ordered_depth', True)
    has_contrastive_head = kwargs.get('has_contrastive_head', False)
    ste_depth = kwargs.get('ste_depth', False)
    if sto_depth:
        return WideResNetStoDepth(28, 10, dropout, num_classes, **kwargs)
    elif dy_depth_width:
        return WideResNetDyDepthWidth(28, 10, dropout, num_classes, init_width_mul=init_width_mul,
                                      ordered_width=ordered_width, ordered_depth=ordered_depth,
                                      has_contrastive_head=has_contrastive_head, ste_depth=ste_depth)
    elif dy_depth:
        return WideResNetDyDepth(28, 10, dropout, num_classes)
    else:
        return WideResNet(28, 10, dropout, num_classes)

# small resnet
def build_resnet_20(num_classes=100, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    if sto_depth:
        return ResNetStoDepth([3, 3, 3], num_classes, **kwargs)
    elif dy_depth:
        return ResNetDyDepth([3, 3, 3], num_classes)
    else:
        return ResNet([3, 3, 3], num_classes)


def build_resnet_32(num_classes=100, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    if sto_depth:
        return ResNetStoDepth([5, 5, 5], num_classes, **kwargs)
    elif dy_depth:
        return ResNetDyDepth([5, 5, 5], num_classes)
    else:
        return ResNet([5, 5, 5], num_classes)


def build_resnet_56(num_classes=100, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    if sto_depth:
        return ResNetStoDepth([9, 9, 9], num_classes, **kwargs)
    elif dy_depth:
        return ResNetDyDepth([9, 9, 9], num_classes)
    else:
        return ResNet([9, 9, 9], num_classes)


def build_resnet_110(num_classes=100, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    if sto_depth:
        return ResNetStoDepth([12, 12, 12], num_classes, **kwargs)
    elif dy_depth:
        return ResNetDyDepth([12, 12, 12], num_classes)
    else:
        return ResNet([12, 12, 12], num_classes)

# The big resnet
def build_resnet_34(num_classes=100, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    dy_depth_width = kwargs.get('dy_depth_width', False)
    ordered_width = kwargs.get('ordered_width', True)
    has_contrastive_head = kwargs.get('has_contrastive_head', False)

    if sto_depth:
        return big_resnet_std.ResNetStoDepth(block=big_resnet_std.BasicBlockStoDepth, layers=[3, 4, 6, 3],
                                             num_classes=num_classes, **kwargs)
    elif dy_depth_width:
        return ResNetDyDepthWidth(block=big_resnet_ddw.BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes,
                                  ordered_width=ordered_width, has_contrastive_head=has_contrastive_head)
    elif dy_depth:
        return big_resnet_ddp.ResNetDyDepth(block=big_resnet_ddp.BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    else:
        return big_resnet.ResNet(block=big_resnet.BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)

def build_resnet_50(num_classes=100, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    dy_depth_width = kwargs.get('dy_depth_width', False)
    ordered_width = kwargs.get('ordered_width', True)
    has_contrastive_head = kwargs.get('has_contrastive_head', False)
    if sto_depth:
        return big_resnet_std.ResNetStoDepth(block=big_resnet_std.BottleneckStoDepth, layers=[3, 4, 6, 3],
                                             num_classes=num_classes, **kwargs)
    elif dy_depth_width:
        return ResNetDyDepthWidth(block=big_resnet_ddw.Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes,
                                  ordered_width=ordered_width, has_contrastive_head=has_contrastive_head)
    elif dy_depth:
        return big_resnet_ddp.ResNetDyDepth(block=big_resnet_ddp.Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
    else:
        return big_resnet.ResNet(block=big_resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
# transformers
def build_vit(num_classes=100, depth=9, **kwargs):
    img_size = kwargs.get('img_size', 32)
    stochastic_depth = kwargs.get('sd', 0)
    patch_size = 4 if img_size == 32 else 8
    return ViTDyDepthWidth(img_size=img_size, patch_size=patch_size, num_classes=num_classes, dim=192,
                mlp_dim_ratio=2, depth=depth, heads=12, dim_head=192 // 12,
                stochastic_depth=stochastic_depth)

def build_swin(num_classes=100, depth=[2, 6, 4], **kwargs):
    img_size = kwargs.get('img_size', 32)
    patch_size = 2 if img_size == 32 else 4
    stochastic_depth = kwargs.get('sd', 0)
    return SwinTransformerDyDepth(img_size=img_size, window_size=4, drop_path_rate=stochastic_depth,
                            patch_size=patch_size, mlp_ratio=2, depths=depth, num_heads=[3, 6, 12],
                            num_classes=num_classes,
                            is_SPT=False, is_LSA=False)

def build_pit(num_classes=100, depth=[2, 6, 4], **kwargs):
    return PiTDyDepth(img_size=32, patch_size=2, num_classes=num_classes, dim=96,
                mlp_dim_ratio=2, depth=depth, heads=(2, 4, 8), dim_head=48,
                stochastic_depth=0, is_SPT=False, is_LSA=False)

def build_t2t(num_classes=100, depth=12, **kwargs):
    return T2T_ViTDyDepth(img_size=32, num_classes=num_classes, drop_path_rate=0, is_SPT=False, is_LSA=False)

def build_cait(num_classes=100, depth=24, **kwargs):
    img_size = kwargs.get('img_size', 32)
    stochastic_depth = kwargs.get('sd', 0)
    patch_size = 4 if img_size == 32 else 8
    return CaiTDyDepth(img_size=img_size, patch_size=patch_size, num_classes=num_classes, stochastic_depth=stochastic_depth,
                is_LSA=False, is_SPT=False)

# mobilenet v3
def build_mobilenet_v3(num_classes, dropout_rate=0.1, **kwargs):
    sto_depth = kwargs.get('sto_depth', False)
    dy_depth = kwargs.get('dy_depth', False)
    dy_depth_width = kwargs.get('dy_depth_width', False)
    ordered_width = kwargs.get('ordered_width', True)
    has_contrastive_head = kwargs.get('has_contrastive_head', False)
    if sto_depth:
        return MobileNetV3StoDepth(num_classes, dropout_rate, **kwargs)
    elif dy_depth_width:
        return MobileNetV3DyDepthWidth(num_classes=num_classes, dropout_rate=dropout_rate, ordered_width=ordered_width,
                                       has_contrastive_head=has_contrastive_head)
    elif dy_depth:
        return MobileNetV3DyDepth(num_classes, dropout_rate)
    else:
        return MobileNetV3(num_classes, dropout_rate)
