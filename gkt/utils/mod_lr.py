import torch
import torchvision.transforms as transforms
import math
from torch.optim.lr_scheduler import _LRScheduler

# import torchvision.transforms.functional as F


# __all__ = ['get_lr_scheduler', 'get_optimizer']


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def get_lr_scheduler(optimizer, trainloader, config_in):
    if config_in['lr_scheduler'] == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[len(trainloader) * s for s in config_in['lr_steps']], gamma=config_in['lr_gamma'])
    elif config_in['lr_scheduler'] == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config_in['epochs'] * len(trainloader))
    elif config_in['lr_scheduler'] == 'warmup_cosine':
        num_steps = int(config_in['epochs'] * len(trainloader))
        warmup_steps = int(10 * len(trainloader))
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_steps,
            cycle_mult=1.,
            max_lr=config_in['lr'],
            min_lr=1e-6,
            warmup_steps=warmup_steps
        )
    else:
        raise NotImplemented('LR scheduler not implemented.')
    return lr_scheduler


def get_optimizer(model, config_in, model_type='typical'):
    if config_in['optimizer'].upper() == 'SGD':
        if model_type == 'typical':
            optimizer = torch.optim.SGD(model.parameters(), lr=config_in['lr'], momentum=config_in['momentum'],
                                        weight_decay=config_in['weight_decay'], nesterov=config_in['nesterov'])
        elif ('mobilenet' in model_type) or (model_type == 'resnet_34') or (model_type == 'resnet_50'):
            param_groups = []
            for name, param in model.named_parameters():
                if 'bn' in name or 'bias' in name:
                    param_groups.append({'params': param, 'weight_decay': 0.})
                else:
                    param_groups.append({'params': param, 'weight_decay': config_in['weight_decay']})
            optimizer = torch.optim.SGD(param_groups, lr=config_in['lr'],
                                        momentum=config_in['momentum'], nesterov=config_in['nesterov'])
    elif config_in['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_in['lr'], weight_decay=config_in['weight_decay'])
    else:
        raise NotImplemented('Optimizer not implemented!')

    return optimizer


# class MyRandomResizedCrop(transforms.RandomResizedCrop):
#
#     def __init__(self, size_list, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.size_list = size_list
#
#     def forward(self, img):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be cropped and resized.
#
#         Returns:
#             PIL Image or Tensor: Randomly cropped and resized image.
#         """
#         random_index = torch.randint(0, len(self.size_list), (1,)).item()
#         self.size = self.size_list[random_index]
#
#         i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         return F.resized_crop(img, i, j, h, w, (self.size, self.size), self.interpolation)
#
#     def __repr__(self):
#         interpolate_str = self.interpolation.value
#         format_string = self.__class__.__name__ + '(size={}'.format(self.size_list)
#         format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
#         format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
#         format_string += ', interpolation={0})'.format(interpolate_str)
#         return format_string





