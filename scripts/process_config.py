"""config utilities for yml file."""
import os
import sys
from pathlib import Path
import yaml
import argparse
import logging
# singletone
FLAGS = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _config_path(name: str) -> str:
    """Resolve a YAML file under the repository ``configs/`` directory."""
    return str(_PROJECT_ROOT / "configs" / name)


class LoaderMeta(type):
    """Constructor for supporting `!include`.
    """
    def __new__(mcs, __name__, __bases__, __dict__):
        """Add include constructer to class."""
        # register the include constructor on the class
        cls = super().__new__(mcs, __name__, __bases__, __dict__)
        cls.add_constructor('!include', cls.construct_include)
        return cls


class Loader(yaml.Loader, metaclass=LoaderMeta):
    """YAML Loader with `!include` constructor.
    """
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

    def construct_include(self, node):
        """Include file referenced at node."""
        filename = os.path.abspath(
            os.path.join(self._root, self.construct_scalar(node)))
        extension = os.path.splitext(filename)[1].lstrip('.')
        with open(filename, 'r') as f:
            if extension in ('yaml', 'yml'):
                return yaml.load(f, Loader)
            else:
                return ''.join(f.readlines())


class AttrDict(dict):
    """Dict as attribute trick.

    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, list):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return.

        """
        yaml_dict = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables.

        """
        ret_str = []
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # treat as AttrDict above
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    """Config with yaml file.

    This class is used to config model hyper-parameters, global constants, and
    other settings with yaml file. All settings in yaml file will be
    automatically logged into file.

    Args:
        filename(str): File name.

    Examples:

        yaml file ``model.yml``::

            NAME: 'neuralgym'
            ALPHA: 1.0
            DATASET: '/mnt/data/imagenet'

        Usage in .py:

        >>> from neuralgym import Config
        >>> config = Config('model.yml')
        >>> print(config.NAME)
            neuralgym
        >>> print(config.ALPHA)
            1.0
        >>> print(config.DATASET)
            /mnt/data/imagenet

    """

    def __init__(self, filename=None, verbose=False):
        assert os.path.exists(filename), 'File {} not exist.'.format(filename)
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, Loader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        super(Config, self).__init__(cfg_dict)
        if verbose:
            print(' pi.cfg '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('USNet')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


# def app():
#     """Load app via stdin from subprocess"""
#     global FLAGS
#     if FLAGS is None:
#         job_yaml_file = None
#         for arg in sys.argv:
#             if arg.startswith('app:'):
#                 job_yaml_file = arg[4:]
#         if job_yaml_file is None:
#             job_yaml_file = sys.stdin.readline()
#
#         FLAGS = Config(job_yaml_file)
#         return FLAGS
#     else:
#         return FLAGS
    # job_yaml_file = 'configs/resnet50_randwidth.yml'
    # FLAGS = Config(job_yaml_file)
    # return FLAGS
def app(arg):
    """Load app via stdin from subprocess"""
    global FLAGS
    FLAGS = Config(arg.configs)

parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=int, default=0)
parser.add_argument('--exp', type=str, default='')
parser.add_argument('--configs', type=str, default=None)
_SUPPORTED_MODELS = (
    'resnet_20', 'resnet_32', 'resnet_56', 'resnet_110', 'resnet_34', 'resnet_50',
    'wide_resnet_16_8', 'wide_resnet_28_10', 'mobilenet_v3',
    'vit', 'vit12', 'vit24', 'swin',
)
parser.add_argument('--model', type=str, default='resnet_32', choices=_SUPPORTED_MODELS)
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--root', type=str, default=None)
parser.add_argument('--dimension', type=str, default='d', choices=['d', 'w', 'dw'])
parser.add_argument('--reset_bn', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--EMA_coef', type=float, default=0.5)
parser.add_argument('--supervision', type=str, choices=['kl', 'part_kl', 'feature', 'gt'], default='kl')
parser.add_argument('--sup_coef', type=float, default=2.0)
parser.add_argument('--main_coef', type=float, default=1.0)
parser.add_argument('--subnet_coef', type=float, default=1.0)
parser.add_argument('--diff_coef', type=float, default=1.0)
parser.add_argument('--subnet_num', type=int, default=1)
parser.add_argument('--sample_version', type=str, default='v1')  # v1/v3
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--dropout_rate', type=float, default=0.0)
parser.add_argument('--sd', type=float, default=0.0)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--multi_head_num', type=int, default=0)
parser.add_argument('--warm_up', type=int, default=0)
parser.add_argument('--structure_num', type=int, default=3)
# others
parser.add_argument('--path', type=str, default='v1')
parser.add_argument('--cluster_interval', type=int, default=1)
parser.add_argument('--head', action='store_true', default=False)
parser.add_argument('--sample_distribution', type=str, default='exp_rise')
parser.add_argument('--mu', type=float, default=0.55)
parser.add_argument('--mu_way', type=str, default='fix')
parser.add_argument('--get_way', type=str, default='norm')
parser.add_argument('--update_way', type=str, default='norm')
parser.add_argument('--oracle', action='store_true', default=False)
parser.add_argument('--cutmixup', action='store_true', default=False)
parser.add_argument('--same_markov', action='store_true', default=False)
parser.add_argument('--every_epoch', action='store_true', default=False)
parser.add_argument('--warm_up_collect', action='store_true', default=False)
parser.add_argument('--markov_prob', type=float, default=0.2)
# distillation temperature (standard KL on softmax(logits / t))
parser.add_argument('--KL_tao', type=float, default=1.0)
parser.add_argument('--min_width', type=float, default=1.0)
parser.add_argument('--min_width_SD', type=str, default='uniform', choices=['uniform', 'linear', 'exp',
                                                                            'cos', 'last_stage', 'couple'])
parser.add_argument('--init_width_mul', type=float, default=1.0)
parser.add_argument('--random_width', action='store_true', default=False)
parser.add_argument('--random_depth', action='store_true', default=False)
parser.add_argument('--ste_depth', action='store_true', default=False)
# for momentum memory bank
parser.add_argument('--MB_type', type=str, default='EMA', choices=['EMA', 'momentum_v1', 'momentum_v2'])
parser.add_argument('--momentum_coef', type=float, default=0.5)

# for contrastive learning
parser.add_argument('--contrastive', action='store_true', default=False)
parser.add_argument('--contrastive_EMA_coef', type=float, default=0.5)
parser.add_argument('--has_contrastive_head', action='store_true', default=False)
parser.add_argument('--contrastive_dim', type=int, default=128)
parser.add_argument('--contrastive_coef', type=float, default=1)
parser.add_argument('--contrastive_tao', type=float, default=0.1)
parser.add_argument('--compare_ratio', type=float, default=1.0, help='the compare ratio of each class for comparison')
parser.add_argument('--update_norm_feature', action='store_true', default=False)
parser.add_argument('--contrastive_loss', type=str, choices=['direct_sum', 'p_sum'], default='p_sum')
parser.add_argument('--contrastive_head_grad_mul', type=float, default=1.0)
parser.add_argument('--debug', action='store_true', default=False)

# for mixup + gkt, different update ways
parser.add_argument('--mixup_type', type=str, choices=['no', 'direct_sub_mixup', 'qp_mixup', 'simple_mixup'], default='no')
# for asyn_mixup
parser.add_argument('--mix_num', type=int, default=3)
parser.add_argument('--main_biggest', action='store_true', default=False)
parser.add_argument('--asyn_mix', action='store_true', default=False)
# for saving epochs
parser.add_argument('--save_epoch', type=str, default=None)


args = parser.parse_args()
args.ordered_width = not args.random_width
args.ordered_depth = not args.random_depth
if args.save_epoch is not None:
    args.save_epoch = eval(args.save_epoch)

if 'cifar' in args.dataset:
    if args.configs is None:
        if (args.model == 'resnet_50') or (args.model == 'resnet_34'):
            args.configs = _config_path('config_cifar_big_multistep.yml')
        elif 'mobilenet' in args.model:
            args.configs = _config_path('config_cifar_big.yml')
        elif args.model == 'vit':
            args.configs = _config_path('config_cifar_vit.yml')
        elif args.model == 'vit12':
            args.configs = _config_path('config_cifar_vit12.yml')
        elif args.model == 'vit24':
            args.configs = _config_path('config_cifar_vit24.yml')
        elif args.model == 'swin':
            args.configs = _config_path('config_cifar_swin.yml')
        else:
            args.configs = _config_path('config_cifar.yml')

    if args.dataset == 'cifar10':
        args.root = args.root if args.root is not None else str(_PROJECT_ROOT / 'data' / 'cifar10')
        args.classes_num = 10
    elif args.dataset == 'cifar100':
        args.root = args.root if args.root is not None else str(_PROJECT_ROOT / 'data' / 'cifar100')
        args.classes_num = 100

elif 'tinyimagenet' in args.dataset:
    if args.configs == None:
        if (args.model == 'resnet_50') or (args.model == 'resnet_34') or ('mobilenet' in args.model):
            args.configs = _config_path('config_tinyimagenet_big.yml')
        elif args.model == 'vit':
            args.configs = _config_path('config_tinyimagenet_vit.yml')
        elif args.model in ('vit12', 'vit24'):
            # No separate YAMLs in repo; share ViT-style schedule on Tiny ImageNet
            args.configs = _config_path('config_tinyimagenet_vit.yml')
        elif args.model == 'swin':
            args.configs = _config_path('config_tinyimagenet_swin.yml')
        else:
            args.configs = _config_path('config_tinyimagenet.yml')
    args.root = args.root if args.root is not None else str(_PROJECT_ROOT / 'data' / 'tinyimagenet')
    # args.root = '/data/dataset/tinyimagenet'
    args.classes_num = 200

args.reset_bn = True
args.dy_depth = args.dimension == 'd'
args.dy_depth_width = args.dimension == 'dw'
args.sto_depth = False

app(args)
