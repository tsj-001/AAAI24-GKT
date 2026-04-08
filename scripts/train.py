import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

import shutil
import time
from functools import reduce
import argparse
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import pickle
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from gkt.model_bank.dynamic_operator import TensorWithMeta
from gkt.utils.sampler import RASampler
from typing import Any, Callable, Optional, Tuple
from scripts.process_config import args, FLAGS, get_logger

import numpy as np
import random
from copy import deepcopy

from gkt.model_bank import *
from gkt.utils import *

# set the GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
torch.cuda.set_device(args.GPU)

# fix the seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
args.seed = seed
# log the time

best_prec1 = 0
total_train_time = 0
train_transform_post = None
test_transform_post = None

logpath = str(_PROJECT_ROOT / 'runs' / 'gkt' / (args.model + '_' + args.exp) / args.dataset / '')
os.makedirs(logpath, exist_ok=True)
logger = get_logger(os.path.join(logpath, 'train.log'))
shutil.copy2(args.configs, os.path.join(logpath, os.path.basename(args.configs)))
with open(os.path.join(logpath, 'args.txt'), 'w') as myfile:
    for k, v in args.__dict__.items():
        myfile.write("{}: {}\n".format(k, v))
shutil.copy2(__file__, os.path.join(logpath, os.path.basename(__file__)))

if 'mobilenet' in args.model or 'resnet_50' in args.model or 'resnet_34' in args.model:
    args.model_type = 'mobilenet'
else:
    args.model_type = 'typical'
with open(os.path.join(logpath, 'rerun.sh'), 'w') as f:
    f.write('nohup python -u ' + reduce(lambda x, y: x + ' ' + y, sys.argv) + ' > log/{}.log 2>&1 &'.format(args.exp))
print('python -u ' + reduce(lambda x, y: x + ' ' + y, sys.argv))


class MyTinyImageNet(datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class MyDataset(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(MyDataset, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class MyDatasetCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(MyDatasetCIFAR10, self).__init__(root, transform=transform,
                                               target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target):
    if len(pred) == 0:
        return 0.0
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    soft_target = label_smooth(target, pred.size(1), label_smoothing)
    return cross_entropy_loss_with_soft_target(pred, soft_target)


class StructureMemoryBank():
    def __init__(self, ema_coef, class_num, data_num, structure_num, is_cuda=True,
                 get_way='norm', update_way='norm', oracle=False, sample_version='v3',
                 warm_up_collect=False, tolerance=0.8, gt_init=True, all_targets=None, gt_init_ls=0):
        self.gt_init = gt_init
        self.gt_init_ls = gt_init_ls

        self.labels = torch.zeros((structure_num * data_num, class_num))
        self.updated_flag = torch.zeros(structure_num * data_num).bool()
        self.all_updated = torch.zeros(structure_num).bool()
        self.targets = torch.zeros(data_num).long()
        self.is_cuda = is_cuda
        if is_cuda:
            self.labels = self.labels.cuda()
            self.updated_flag = self.updated_flag.cuda()
            self.all_updated = self.all_updated.cuda()
            self.targets = self.targets.cuda()
        if warm_up_collect:
            self.warm_up_labels = torch.zeros((data_num, class_num))
            if is_cuda:
                self.warm_up_labels = self.warm_up_labels.cuda()
        if gt_init:
            self.labels.scatter_(dim=1, index=all_targets[:, None].repeat(structure_num, 1),
                                 src=torch.ones_like(self.labels))
            self.updated_flag[:] = True
            self.all_updated[:] = True
            self.targets = all_targets
            self.labels = (1 - self.gt_init_ls) * self.labels + self.gt_init_ls / class_num
            if warm_up_collect:
                self.warm_up_labels.scatter_(dim=1, index=all_targets[:, None],
                                             src=torch.ones_like(self.warm_up_labels))
        self.ema_coef = ema_coef
        self.data_num = data_num
        self.class_num = class_num
        self.structure_num = structure_num
        self.get_way = get_way
        self.update_way = update_way
        self.oracle = oracle
        self.sample_version = sample_version
        self.warm_up_collect = warm_up_collect
        self.tolerance = tolerance

    def update(self, labels, indices, targets, structure_id, chosen_depth, is_warm_up):
        if not self.is_cuda:
            labels = labels.cpu()
            indices = indices.cpu()
            targets = targets.cpu()
        if not is_warm_up:
            map_indices = (structure_id * self.data_num + indices).long()  # [128]
            has_updated = self.updated_flag[map_indices]
            if self.oracle:
                is_correct = labels.argmax(dim=1) == targets
                new_update = is_correct & (~has_updated)
                has_updated = has_updated & is_correct  # [128] mask
                new_indices = map_indices[new_update]
            else:
                new_indices = map_indices[~has_updated]

            updated_indices = map_indices[has_updated]
            updated_targets = targets[has_updated]
            if self.update_way == 'norm':
                self.labels[updated_indices] = (1 - self.ema_coef) * self.labels[updated_indices] \
                                               + self.ema_coef * labels[has_updated]
            elif self.update_way == 'correct_update':
                tolerance = 0.2
                max_update = 0.8
                memory_prob = torch.gather(input=self.labels[updated_indices], index=updated_targets[:, None],
                                           dim=1).squeeze(dim=1)
                labels_prob = torch.gather(input=labels[has_updated], index=updated_targets[:, None], dim=1).squeeze(
                    dim=1)  # [128]
                diff_prob = labels_prob - memory_prob
                buffer = 1 + (tolerance - 1) * memory_prob
                EMA_coef = max_update / buffer * diff_prob + tolerance * memory_prob * max_update / buffer
                EMA_coef[EMA_coef < 0] = 0
                self.labels[updated_indices] = (1 - EMA_coef)[:, None] * self.labels[updated_indices] \
                                               + EMA_coef[:, None] * labels[has_updated]
            elif self.update_way == 'oracle_EMA':
                labels_prob = torch.gather(input=labels[has_updated], index=updated_targets[:, None], dim=1).squeeze(
                    dim=1)  # [128]
                EMA_coef = 1 - (-labels_prob).exp()
                self.labels[updated_indices] = (1 - EMA_coef)[:, None] * self.labels[updated_indices] \
                                               + EMA_coef[:, None] * labels[has_updated]
            elif self.update_way == 'frequency_update':
                if self.sample_version == 'v1':
                    update_EMA_coef_list = [0.1, 0.2, 0.5, 0.6]
                else:
                    raise ValueError
                EMA_coef = update_EMA_coef_list[chosen_depth[structure_id] - 1]
                self.labels[updated_indices] = (1 - EMA_coef) * self.labels[updated_indices] \
                                               + EMA_coef * labels[has_updated]
            elif self.update_way == 'gt_weight_mean':
                tolerance = self.tolerance
                if len(updated_indices) != 0:
                    pass
                memory_prob = torch.gather(input=self.labels[updated_indices], index=updated_targets[:, None],
                                           dim=1).squeeze(dim=1)
                labels_prob = torch.gather(input=labels[has_updated], index=updated_targets[:, None], dim=1).squeeze(
                    dim=1)  # [128]
                hard_gt = torch.zeros_like(labels[has_updated])
                hard_gt.scatter_(dim=1, index=updated_targets[:, None], src=torch.ones_like(labels[has_updated]))
                weight_labels = ((1 - tolerance) * memory_prob - 1) / (labels_prob - 1)  # [128]
                weight_labels[weight_labels < 0] = 0
                weight_labels[weight_labels > 1] = 1
                weight_labels = weight_labels[:, None]
                corrected_label = weight_labels * labels[has_updated] + (1 - weight_labels) * hard_gt
                self.labels[updated_indices] = (1 - self.ema_coef) * self.labels[updated_indices] \
                                               + self.ema_coef * corrected_label
            else:
                raise ValueError
            if self.oracle:
                self.labels[new_indices] = labels[new_update]
            else:
                self.labels[new_indices] = labels[~has_updated]
            self.targets[indices] = targets
            if self.oracle:
                self.updated_flag[new_indices] = True
            else:
                self.updated_flag[map_indices] = True
        elif self.warm_up_collect:
            has_updated = (self.warm_up_labels[indices].sum(dim=1) != 0)
            indices_has_updated = indices[has_updated]
            self.warm_up_labels[indices_has_updated] = (1 - self.ema_coef) * self.warm_up_labels[indices_has_updated] \
                                                       + self.ema_coef * labels[has_updated]
            self.warm_up_labels[indices[~has_updated]] = labels[~has_updated]

    def get_label(self, indices, labels, targets, structure_id):
        if not self.is_cuda:
            source_device = labels.device
            indices = indices.cpu()
            labels = labels.cpu()
            targets = targets.cpu()
        if self.get_way == 'norm':
            has_get = torch.zeros_like(targets).bool()
            if structure_id != (self.structure_num - 1):
                map_indices = ((structure_id + 1) * self.data_num + indices).long()  # [128]
            else:
                map_indices = (structure_id * self.data_num + indices).long()  # [128]
            sup_memory = torch.zeros_like(labels)
            # get the correct indices
            is_updated = self.updated_flag[map_indices]  # [128] mask
            updated_indices = map_indices[is_updated]
            sup_memory[is_updated] = self.labels[updated_indices]
            has_get[is_updated] = True  # [128] mask
            # for non updated indices, try to find one
            non_updated_indices = map_indices[~is_updated]
            if (~is_updated).sum() != 0:
                # from small to large
                for i in range(self.structure_num):
                    try_map_indices = (i * self.data_num + indices).long()
                    try_is_updated = self.updated_flag[try_map_indices]
                    try_is_updated_and_not_updated = try_is_updated & (~is_updated)
                    try_updated_indices = try_map_indices[try_is_updated_and_not_updated]
                    sup_memory[try_is_updated_and_not_updated] = self.labels[try_updated_indices]
                    has_get[try_is_updated_and_not_updated] = True
            # can't find in any memory bank
            keep_mask = ~has_get
            sup_memory[keep_mask] = labels[keep_mask]
            if not self.is_cuda:
                sup_memory = sup_memory.to(source_device)
            return sup_memory
        elif self.get_way == 'avg_ens':
            has_get = torch.zeros(int(self.structure_num * targets.size(0)), device=targets.device).bool()  # [3*128]
            map_indices = ((torch.arange(0, self.structure_num, device=targets.device) * self.data_num)
                           [:, None].expand(-1, targets.size(0)) + indices[None, :]).view(-1)  # [3*128]
            # sup_memory = torch.zeros_like(labels)
            sup_memory_ori = torch.zeros((self.structure_num * labels.size(0), labels.size(1)),
                                         device=labels.device)  # [3*128,100]
            # get the correct indices
            is_updated = self.updated_flag[map_indices]  # [3*128] mask
            updated_indices = map_indices[is_updated]
            sup_memory_ori[is_updated] = self.labels[updated_indices]
            has_get[is_updated] = True  # [3*128] mask
            sup_memory = sup_memory_ori.view(self.structure_num, targets.size(0), -1)[structure_id:].mean(dim=0)
            has_get = has_get.view(self.structure_num, targets.size(0))[structure_id:]  # [k, 128]
            has_get = has_get.float().sum(dim=0).bool()
            keep_mask = ~has_get
            sup_memory[keep_mask] = labels[keep_mask]
            return sup_memory
        elif self.get_way == 'correct_ens':
            has_get = torch.zeros(int(self.structure_num * targets.size(0)), device=targets.device).bool()  # [3*128]
            map_indices = ((torch.arange(0, self.structure_num, device=targets.device) * self.data_num)
                           [:, None].expand(-1, targets.size(0)) + indices[None, :]).view(-1)  # [3*128]
            # sup_memory = torch.zeros_like(labels)
            sup_memory_ori = torch.zeros((self.structure_num * labels.size(0), labels.size(1)),
                                         device=labels.device)  # [3*128,100]
            # get the correct indices
            is_updated = self.updated_flag[map_indices]  # [3*128] mask
            updated_indices = map_indices[is_updated]
            sup_memory_ori[is_updated] = self.labels[updated_indices]
            has_get[is_updated] = True  # [3*128] mask
            # correct weighted
            sup_memory_ori_target_prob = torch.gather(
                input=sup_memory_ori.view(self.structure_num, targets.size(0), -1),
                index=targets[None, :, None].expand(self.structure_num, -1, -1), dim=2)  # [3,128,1]

            sup_memory_ori_weight = sup_memory_ori_target_prob[structure_id:] / (
                    sup_memory_ori_target_prob[structure_id:].sum(dim=0, keepdim=True) + 1e-12)  # [k,128,1]
            sup_memory = (sup_memory_ori_weight * sup_memory_ori.view(self.structure_num, targets.size(0), -1)[
                                                  structure_id:]).sum(dim=0)
            has_get = has_get.view(self.structure_num, targets.size(0))[structure_id:]  # [k, 128]
            has_get = has_get.float().sum(dim=0).bool()
            keep_mask = ~has_get
            sup_memory[keep_mask] = labels[keep_mask]
            return sup_memory
        elif self.get_way == 'self_and_larger_ens':
            self_has_get = torch.zeros_like(targets).bool()
            larger_has_get = torch.zeros_like(targets).bool()
            if structure_id != (self.structure_num - 1):
                map_indices = ((structure_id + 1) * self.data_num + indices).long()  # [128]
            else:
                map_indices = (structure_id * self.data_num + indices).long()  # [128]
        elif self.get_way == 'self':
            has_get = torch.zeros_like(targets).bool()
            map_indices = (structure_id * self.data_num + indices).long()  # [128]
            sup_memory = torch.zeros_like(labels)
            # get the correct indices
            is_updated = self.updated_flag[map_indices]  # [128] mask
            updated_indices = map_indices[is_updated]
            sup_memory[is_updated] = self.labels[updated_indices]
            has_get[is_updated] = True  # [128] mask
            # for non updated indices, try to find one
            non_updated_indices = map_indices[~is_updated]
            if (~is_updated).sum() != 0:
                # from small to large
                for i in range(self.structure_num):
                    try_map_indices = (i * self.data_num + indices).long()
                    try_is_updated = self.updated_flag[try_map_indices]
                    try_is_updated_and_not_updated = try_is_updated & (~is_updated)
                    try_updated_indices = try_map_indices[try_is_updated_and_not_updated]
                    sup_memory[try_is_updated_and_not_updated] = self.labels[try_updated_indices]
                    has_get[try_is_updated_and_not_updated] = True
            # can't find in any memory bank
            keep_mask = ~has_get
            sup_memory[keep_mask] = labels[keep_mask]
            return sup_memory
        else:
            raise ValueError

    def set_ema_coef(self, ema_coef):
        self.ema_coef = ema_coef

    def allocate_memory(self):
        self.labels[:] = self.warm_up_labels.repeat(self.structure_num, 1)


class STinSTSampler():
    # sampling distribution: SD
    def __init__(self, stage_list=(4, 4, 4), sample_num=3, sample_version='v1', markov_prob=0.5,
                 head=False, head_p=0.5, SD='uniform', ordered=True, same_markov=False, dimension='d', min_width=0.8,
                 init_width_mul=1.0, min_width_SD='uniform'):
        self.stage_list = stage_list
        self.full_width = len(stage_list) * [init_width_mul]
        self.stage_num = len(self.stage_list)
        self.sample_num = sample_num
        self.now_sample = self.sample_num - 1
        self.before_sample = self.sample_num - 1
        # save the before architectures
        self.before_sample_st = list(stage_list)
        self.before_sample_width = len(stage_list) * [1.0]
        self.sample_version = sample_version
        self.head = head
        self.head_p = head_p
        self.SD = SD
        self.markov_prob = markov_prob
        self.same_markov = same_markov
        self.dimension = dimension
        self.min_width = min_width
        self.init_width_mul = init_width_mul
        if self.same_markov:
            self.base_prob_list = self.stage_num * [self.markov_prob]
        else:
            self.base_prob_list = (
                    (np.ones(self.stage_num) * self.markov_prob) ** np.arange(1, self.stage_num + 1)[::-1]).tolist()
        self.ordered = ordered

        # for width
        self.min_width_SD = min_width_SD
        if self.min_width_SD == 'uniform':
            self.min_width_SD_list = [self.min_width] * len(stage_list)
        elif self.min_width_SD == 'linear' or self.min_width_SD == 'couple':
            self.min_width_SD_list = np.linspace(1.0, self.min_width, len(stage_list)+1)[1:].tolist()
        elif self.min_width_SD == 'exp':
            alpha = math.e ** (math.log(self.min_width) / 3)
            self.min_width_SD_list = (alpha ** np.arange(len(stage_list+1)))[1:]
        elif self.min_width_SD == 'cos':
            self.min_width_SD_list = np.cos(np.arange(len(stage_list)+1) * math.acos(self.min_width) / len(stage_list))[1:].tolist()
        elif self.min_width_SD == 'last_stage':
            self.min_width_SD_list = [1.0] * (len(stage_list) - 1) + [self.min_width]
        else:
            raise NotImplementedError

    def shuffle_layers(self, ordered_depth):
        assert not self.ordered
        random_depth_list = [[1] + (d - 1) * [0] for d in self.stage_list]
        for i, depth in enumerate(ordered_depth):
            chosen_index = np.random.choice(range(1, self.stage_list[i]), depth - 1, replace=False)
            temp_list = np.array(random_depth_list[i])
            temp_list[chosen_index] = 1
            random_depth_list[i] = temp_list.tolist()
        return random_depth_list

    def sample(self, epoch):
        depth_candi_ad = []
        for d in self.before_sample_st:
            depth_candi_ad.append(list(range(d)))
        depth_st = self.before_sample_st
        width_st = []
        for i in range(self.stage_num):
            if self.SD == 'uniform':
                sample_distribution = np.ones(len(depth_candi_ad[i]))
                sample_distribution /= sample_distribution.sum()
            elif self.SD == 'markov':
                base_prob = self.base_prob_list[i]
                n = len(depth_candi_ad[i])
                sample_distribution = np.ones(n) * base_prob
                sample_distribution = sample_distribution ** np.arange(1, n + 1)
                sample_distribution /= sample_distribution.sum()
            elif self.SD == 'linear_rise':
                min_prob = 0
                n = len(depth_candi_ad[i])
                sample_distribution = np.linspace(min_prob, (2 - n * min_prob) / n, n)[::-1]
                sample_distribution /= sample_distribution.sum()
            else:
                raise ValueError
            depth_st[i] -= np.random.choice(depth_candi_ad[i], size=1, p=sample_distribution)[0]
            # sample the width
            min_width_i = self.min_width_SD_list[i]
            if not self.min_width_SD == 'couple':
                min_width, max_width \
                    = min_width_i + self.now_sample * (self.full_width[i] - min_width_i) / self.sample_num, \
                      self.before_sample_width[i]
                width_st.append(min_width + (max_width - min_width) * np.random.rand())
            else:
                width_st.append(np.linspace(min_width_i, 1.0, self.stage_list[i])[int(depth_st[i]-1)])
        self.before_sample_st = depth_st
        self.before_sample_width = width_st
        self.before_sample = self.now_sample
        self.now_sample -= 1
        if self.now_sample < 0:
            # begin to loop
            self.before_sample_st = list(self.stage_list)
            self.now_sample = self.sample_num - 1
            self.before_sample_width = self.full_width
        if not self.ordered:
            depth_result = self.shuffle_layers(depth_st)
        else:
            depth_result = depth_st

        if self.dimension == 'd':
            return depth_result
        elif self.dimension == 'dw':
            return [depth_result, width_st]


def make_divisible(v, divisor=1, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def recover_model(model, my_args=None):
    # there are many flags in the network to control the action, and this function aims to recover all the flags to
    # make model be main network mode
    # if FLAGS['model'] == 'wideresnet':
    #     if args.dimension == 'd':
    if args.dimension == 'd':
        model.module.set_running_depth(my_args.depth_list)
    elif args.dimension == 'dw':
        model.module.set_running_depth([my_args.depth_list, len(my_args.depth_list) * [1.0]])
    else:
        raise NotImplementedError


def set_running_statistics(model, data_loader):
    # use 2000 image in train loader to reset the bn layer
    bn_mean = {}
    bn_var = {}
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    forward_model = copy.deepcopy(model.module)
    # forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    if isinstance(x, TensorWithMeta):
                        x = x.input
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x, batch_mean, batch_var, bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim], False,
                        0.0, bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])
    cnt = 0
    with torch.no_grad():
        # DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        forward_model.train()
        for images, labels, indices in data_loader:
            images = images.cuda()
            forward_model(images)
            cnt += images.size(0)
            if cnt >= 4000:
                break
        forward_model.eval()
        # DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if (name in bn_mean and bn_mean[name].count > 0) or (name[7:] in bn_mean and bn_mean[name[7:]].count > 0):
            if name[7:] in bn_mean and bn_mean[name[7:]].count > 0:
                name = name[7:]
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

def main():
    global best_prec1
    global train_transform_post
    global test_transform_post

    if 'cifar' in args.dataset:
        # Data loading code
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    elif 'tinyimagenet' in args.dataset:
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        transform_test = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), normalize]
        )

    if 'mobilenet' in args.model or 'resnet_50' in args.model or 'resnet_34' in args.model:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(160, 160), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                normalize
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(int(math.ceil(160 / 0.875))),
                transforms.CenterCrop(160),
                transforms.ToTensor(),
                normalize
            ]
        )

    if 'vit' in args.model or 'swin' in args.model:
        from gkt.utils.autoaug import CIFAR10Policy
        from gkt.utils.random_erasing import RandomErasing
        normalize = transforms.Normalize(mean=(0.507, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                normalize,
                RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=(0.507, 0.4865, 0.4409))
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize
            ]
        )

    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'tinyimagenet')

    kwargs = {'num_workers': FLAGS.workers, 'pin_memory': True}
    if args.dataset == 'cifar10':
        train_dataset = MyDatasetCIFAR10(root=args.root, train=True, download=True, transform=transform_train)
        val_dataset = datasets.__dict__[args.dataset.upper()](args.root, train=False, transform=transform_test)
    elif args.dataset == 'cifar100':
        train_dataset = MyDataset(args.root, train=True, download=True, transform=transform_train)
        val_dataset = datasets.__dict__[args.dataset.upper()](args.root, train=False, transform=transform_test)
    elif args.dataset == 'tinyimagenet':
        train_dataset = MyTinyImageNet(os.path.join(args.root, 'train'), transform=transform_train)
        val_dataset = datasets.ImageFolder(os.path.join(args.root, 'val'), transform=transform_test)
    if 'vit' in args.model or 'swin' in args.model:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=FLAGS.workers, pin_memory=True,
            batch_sampler=RASampler(len(train_dataset), FLAGS.batch_size, 1, 3, shuffle=True, drop_last=True))
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, **kwargs)

    # create model
    if args.model == 'resnet_20':
        model = build_resnet_20(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth)
        args.depth_list = [3, 3, 3]
    elif args.model == 'resnet_32':
        model = build_resnet_32(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth)
        args.depth_list = [5, 5, 5]
    elif args.model == 'resnet_56':
        model = build_resnet_56(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth)
        args.depth_list = [9, 9, 9]
    elif args.model == 'resnet_110':
        model = build_resnet_110(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth,
                                 dy_depth_width=args.dy_depth_width)
        args.depth_list = [12, 12, 12]
    elif args.model == 'resnet_50':
        model = build_resnet_50(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth,
                                dy_depth_width=args.dy_depth_width, dropout_rate=args.dropout_rate)
        args.depth_list = [3, 4, 6, 3]
    elif args.model == 'resnet_34':
        model = build_resnet_34(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth,
                                dy_depth_width=args.dy_depth_width, dropout_rate=args.dropout_rate)
        args.depth_list = [3, 4, 6, 3]
    elif args.model == 'wide_resnet_16_8':
        model = build_wide_resnet_16_8(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth,
                                       dy_depth_width=args.dy_depth_width)
        args.depth_list = [2, 2, 2]
    elif args.model == 'wide_resnet_28_10':
        model = build_wide_resnet_28_10(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth,
                                        dy_depth_width=args.dy_depth_width, init_width_mul=args.init_width_mul,
                                        ordered_width=args.ordered_width, ordered_depth=args.ordered_depth,
                                        ste_depth=args.ste_depth)
        args.depth_list = [4, 4, 4]
    elif args.model == 'mobilenet_v3':
        model = build_mobilenet_v3(num_classes=args.classes_num, sto_depth=False, dy_depth=args.dy_depth,
                                   dy_depth_width=args.dy_depth_width, dropout_rate=args.dropout_rate)
        args.depth_list = [2, 3, 4, 2, 3]
    elif args.model == 'vit':
        model = build_vit(num_classes=args.classes_num, depth=9)
        args.depth_list = [3, 3, 3]
    elif args.model == 'vit12':
        model = build_vit(num_classes=args.classes_num, depth=12)
        args.depth_list = [4, 4, 4]
    elif args.model == 'vit24':
        model = build_vit(num_classes=args.classes_num, depth=24)
        args.depth_list = [8, 8, 8]
    elif args.model == 'swin':
        model = build_swin(num_classes=args.classes_num, depth=[2, 6, 4])
        args.depth_list = [2, 6, 4]
    else:
        raise NotImplementedError

    if args.dy_depth or args.dy_depth_width:
        args.depth_list = deepcopy(model.depth)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model, device_ids=[args.GPU, ], output_device=args.GPU).cuda()
    # torch_dist.init_process_group(backend='nccl', init_method='env://')
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.GPU, ], output_device=args.GPU)
    # optionally resume from a checkpoint
    if FLAGS.resume:
        if os.path.isfile(FLAGS.resume):
            print("=> loading checkpoint '{}'".format(FLAGS.resume))
            checkpoint = torch.load(FLAGS.resume)
            FLAGS.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(FLAGS.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(FLAGS.resume))

    # cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(model, FLAGS, args.model_type)

    # cosine learning rate
    scheduler = get_lr_scheduler(optimizer, train_loader, FLAGS)

    label_memory_bank = StructureMemoryBank(
        ema_coef=args.EMA_coef, class_num=args.classes_num,
        data_num=len(train_dataset), structure_num=args.structure_num,
        is_cuda=True,
        get_way=args.get_way, update_way=args.update_way, oracle=args.oracle,
        sample_version=args.sample_version, warm_up_collect=args.warm_up_collect, tolerance=0.8,
        all_targets=torch.tensor(train_dataset.targets), gt_init=False,
        gt_init_ls=False)

    sampler = STinSTSampler(
        stage_list=args.depth_list, sample_num=3, sample_version=args.sample_version,
        head=args.head, SD='markov', markov_prob=args.markov_prob,
        ordered=args.ordered_depth, same_markov=args.same_markov, dimension=args.dimension,
        min_width=args.min_width,
        init_width_mul=args.init_width_mul, min_width_SD=args.min_width_SD)
    # initialize the subnet.txt
    with open(os.path.join(logpath, 'subnet.txt'), 'a') as myfile:
        myfile.write('')
    for epoch in range(FLAGS.start_epoch, FLAGS.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, label_memory_bank, sampler)

        if args.reset_bn:
            set_running_statistics(model, train_loader)
        prec1, prec5 = validate(val_loader, model, criterion, epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            best_prec5 = prec5
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        with open(os.path.join(logpath, 'time.txt'), 'w') as myfile:
            myfile.write('Epoch:{} total time:{}'.format(epoch, total_train_time))
    logger.info('Best accuracy: Top1:{} Top5:{}'.format(best_prec1, best_prec5))


def train(train_loader, model, criterion, optimizer, scheduler, epoch, label_memory_bank, sampler):
    """单轮训练：GKT（结构记忆库 + 子结构采样 + 跨结构知识蒸馏）。"""
    global total_train_time
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target, indices) in enumerate(train_loader):
        data_time.update(time.time() - end)
        train_time = time.time()
        indices = indices.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()

        resolution = FLAGS.resos[random.randint(0, len(FLAGS.resos) - 1)]
        chosen_depth = sampler.sample(epoch)
        with open(os.path.join(logpath, 'subnet.txt'), 'a') as myfile:
            myfile.write('iter: {}, memory_bank: {}, subnet: {}\n'.format(
                epoch * len(train_loader) + i, sampler.before_sample, chosen_depth))
        model.module.set_running_depth(chosen_depth)
        out = model(F.interpolate(input, (resolution, resolution), mode='bilinear', align_corners=True))
        prec1, prec5 = accuracy(out.detach().data, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        main_coef = args.main_coef

        if args.label_smoothing == 0:
            main_loss = main_coef * criterion(out, target)
        else:
            main_loss = main_coef * cross_entropy_with_label_smoothing(out, target, args.label_smoothing)
        out_soft = F.softmax(out.detach() / args.KL_tao, dim=1)

        if epoch >= args.warm_up:
            sup_coef = args.sup_coef
            teacher_out = label_memory_bank.get_label(indices=indices, labels=out_soft, targets=target,
                                                      structure_id=sampler.before_sample)
            loss = sup_coef * torch.nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(out / args.KL_tao, dim=1), teacher_out)
            sup_losses.update(loss.data.item(), input.size(0))
            (main_loss + loss).backward()
        else:
            main_loss.backward()
        losses.update(main_loss.data.item(), input.size(0))
        label_memory_bank.update(labels=out_soft, indices=indices,
                                 targets=target, structure_id=sampler.before_sample,
                                 chosen_depth=chosen_depth, is_warm_up=epoch < args.warm_up)

        if args.dy_depth or args.dy_depth_width:
            recover_model(model, my_args=args)
        # compute gradient and do SGD step
        optimizer.step()
        scheduler.step()
        # measure elapsed time
        total_train_time += (time.time() - train_time)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % FLAGS.print_freq == 0:
            # if i % 10 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'LR:{3: .4f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Sup_Loss {sup_loss.val:.4f} ({sup_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time,
                data_time=data_time,
                loss=losses, sup_loss=sup_losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    global test_transform_post

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # if test_transform_post is not None:
        #     input = test_transform_post(input)

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % FLAGS.print_freq == 0:
            logger.info('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                        'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, FLAGS.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    logger.info(
        '* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
            epoch, FLAGS.epochs, top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = logpath
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(logpath, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
