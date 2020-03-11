import os
from morphx.data import basics
from morphx.processing import clouds


class ArgsContainer(object):
    def __init__(self,
                 save_root: str,
                 train_path: str,
                 sample_num: int,
                 name: str,
                 class_num: int,
                 train_transforms: list,
                 batch_size: int,
                 input_channels: int,
                 features: dict,
                 max_step_size: int = 500000,
                 density_mode: bool = True,
                 chunk_size: int = None,
                 tech_density: int = None,
                 bio_density: int = None,
                 use_val: bool = True,
                 val_transforms: list = None,
                 val_path: str = None,
                 val_iter: int = 1,
                 val_freq: int = 20,
                 track_running_stats: bool = False,
                 use_big: bool = True,
                 random_seed: int = 0,
                 use_cuda: bool = True,
                 ):

        self._save_root = os.path.expanduser(save_root)
        self._train_path = os.path.expanduser(train_path)
        self._name = name
        self._val_save_path = f'{self._save_root}validation/{name}/'
        self._val_info_path = f'{self._val_save_path}info/'
        self._train_save_path = f'{self._save_root}{name}/'

        self._class_num = class_num
        self._sample_num = sample_num
        self._batch_size = batch_size

        self._train_transforms = train_transforms
        if self._train_transforms is None:
            self._train_transforms = [clouds.Identity()]

        self._input_channels = input_channels
        self._features = features
        for key in self._features.keys():
            if len(self._features[key]) != self._input_channels:
                raise ValueError("Feature dimension doesn't match with number of input channels.")

        self._density_mode = density_mode
        self._chunk_size = chunk_size
        self._tech_density = tech_density
        self._bio_density = bio_density
        if self._density_mode and (self._bio_density is None or self._tech_density is None):
            raise ValueError("In density mode both bio and tech densities are required.")
        if not self._density_mode and self._chunk_size is None:
            raise ValueError("In context mode, the size of the chunks is required.")

        self._use_val = use_val
        self._val_path = val_path
        self._val_iter = val_iter
        self._val_freq = val_freq
        self._val_transforms = val_transforms
        if self._val_transforms is None:
            self._val_transforms = [clouds.Identity()]

        if len(self._val_transforms) > len(self._train_transforms):
            raise ValueError("Validation has different transformations than training.")
        for transform in self._train_transforms:
            valid = False
            if isinstance(transform, clouds.Normalization):
                for val_transform in self._val_transforms:
                    if isinstance(val_transform, clouds.Normalization):
                        if valid:
                            raise ValueError("Validation has multiple normalization transforms.")
                        if val_transform.radius == transform.radius:
                            valid = True
                if not valid:
                    raise ValueError("Validation has normalization which does not match with the normalization"
                                     "during training.")
            elif isinstance(transform, clouds.Center):
                for val_transform in self._val_transforms:
                    if isinstance(val_transform, clouds.Center):
                        valid = True
                if not valid:
                    raise ValueError("Validation transforms have no Center transform, but training has.")

        self._use_cuda = use_cuda
        self._use_big = use_big
        self._random_seed = random_seed
        self._track_running_stats = track_running_stats
        self._max_step_size = max_step_size

    @property
    def normalization(self):
        for transform in self._train_transforms:
            if isinstance(transform, clouds.Normalization):
                return transform.radius

    @property
    def save_root(self):
        return self._save_root

    @property
    def train_path(self):
        return self._train_path

    @property
    def val_save_path(self):
        return self._val_save_path

    @property
    def train_save_path(self):
        return self._train_save_path

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def sample_num(self):
        return self._sample_num

    @property
    def name(self):
        return self._name

    @property
    def class_num(self):
        return self._class_num

    @property
    def train_transforms(self):
        return self._train_transforms

    @property
    def val_transforms(self):
        return self._val_transforms

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def random_seed(self):
        return self._random_seed

    @property
    def val_path(self):
        return self._val_path

    @property
    def track_running_stats(self):
        return self._track_running_stats

    @property
    def use_val(self):
        return self._use_val

    @property
    def features(self):
        return self._features

    @property
    def tech_density(self):
        return self._tech_density

    @property
    def bio_density(self):
        return self._bio_density

    @property
    def density_mode(self):
        return self._density_mode

    @property
    def val_iter(self):
        return self._val_iter

    @property
    def val_freq(self):
        return self._val_freq

    @property
    def max_step_size(self):
        return self._max_step_size

    @property
    def use_big(self):
        return self._use_big

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def val_info_path(self):
        if not os.path.exists(self._val_info_path):
            os.makedirs(self._val_info_path)
        return self._val_info_path

    def info2pkl(self, path: str):
        attr_dict = {'density_mode': self._density_mode, 'chunk_size': self._chunk_size,
                     'tech_density': self._tech_density, 'bio_density': self._bio_density,
                     'sample_num': self._sample_num}
        basics.save2pkl(attr_dict, path, name='argscont')


def pkl2container(file: str) -> ArgsContainer:
    args = basics.load_pkl(file)
    slashs = [pos for pos, char in enumerate(file) if char == '/']
    current_save_root = file[:slashs[-2]+1]

    return ArgsContainer(save_root=current_save_root, train_path=args[1], chunk_size=args[2], sample_num=args[3],
                         name=args[4], class_num=args[5], train_transforms=args[6], val_transforms=args[7],
                         batch_size=args[8], use_cuda=args[9], input_channels=args[10], use_big=args[11],
                         random_seed=args[12], val_path=args[13], track_running_stats=args[14], use_val=args[15],
                         features=args[16], tech_density=args[17], bio_density=args[18], density_mode=args[19],
                         val_iter=args[20], val_freq=args[21], max_step_size=args[23])
