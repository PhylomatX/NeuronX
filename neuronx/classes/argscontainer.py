import os
import numpy as np
import pickle
import torch
from typing import List, Tuple, Union, Optional
from morphx.data import basics
from morphx.processing import clouds
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset


class ArgsContainer(object):
    def __init__(self,
                 save_root: str = None,
                 train_path: Union[str, SuperSegmentationDataset] = None,
                 sample_num: int = None,
                 name: str = None,
                 class_num: int = None,
                 train_transforms: List[clouds.Transformation] = None,
                 batch_size: int = None,
                 input_channels: int = None,
                 features: dict = None,
                 max_step_size: int = 500000,
                 density_mode: bool = True,
                 chunk_size: int = None,
                 tech_density: int = None,
                 bio_density: int = None,
                 use_val: bool = True,
                 val_transforms: List[clouds.Transformation] = None,
                 val_path: str = None,
                 val_iter: int = 1,
                 val_freq: int = 20,
                 track_running_stats: bool = False,
                 use_big: bool = True,
                 random_seed: int = 0,
                 use_cuda: bool = True,
                 label_mappings: List[Tuple[int, int]] = None,
                 hybrid_mode: bool = False,
                 optimizer: str = 'adam',
                 scheduler: str = 'steplr',
                 splitting_redundancy: int = 1,
                 use_norm: bool = True,
                 class_weights: Union[str, np.ndarray] = None,
                 use_bias: bool = False,
                 norm_type: str = 'bn',
                 label_remove: List[int] = None,
                 sampling: bool = True,
                 batch_avg: int = None,
                 kernel_size: int = 16,
                 neighbor_nums: List[int] = None,
                 dilations: List[int] = None,
                 reductions: List[int] = None,
                 first_layer: bool = True,
                 padding: int = None,
                 centroids: bool = False,
                 nn_center: bool = True,
                 optim_kernels: bool = True,
                 pl: int = 64,
                 dropout: float = 0,
                 split_on_demand: bool = False,
                 split_jitter: int = 0,
                 cp_norm: bool = False,
                 architecture: List[tuple] = None,
                 act: str = None,
                 epoch_size: int = None,
                 workers: int = 2,
                 voxel_sizes: Optional[dict] = None,
                 ssd_exclude: List[int] = None,
                 ssd_include: List[int] = None,
                 ssd_labels: str = None,
                 exclude_borders: int = 0,
                 rebalance: dict = None):

        if save_root is not None:
            self._save_root = os.path.expanduser(save_root)
            self._train_save_path = f'{self._save_root}{name}/'
        else:
            self._save_root = None
            self._train_save_path = None
        if type(train_path) != SuperSegmentationDataset and train_path is not None:
            self._train_path = os.path.expanduser(train_path)
        else:
            self._train_path = train_path
        self._name = name
        self._class_num = class_num
        self._sample_num = sample_num
        self._batch_size = batch_size
        self._train_transforms = train_transforms
        if self._train_transforms is None:
            self._train_transforms = [clouds.Identity()]
        self._input_channels = input_channels
        self._features = features
        if self._features is not None:
            for key in self._features.keys():
                if isinstance(self._features[key], dict):
                    for item in self._features[key]:
                        if len(self._features[key][item]) != self._input_channels:
                            raise ValueError("Feature dimension doesn't match with number of input channels.")
                elif isinstance(self._features[key], int):
                    if self._input_channels != 1:
                        raise ValueError("Feature dimension doesn't match with number of input channels.")
                elif len(self._features[key]) != self._input_channels:
                    raise ValueError("Feature dimension doesn't match with number of input channels.")
        self._density_mode = density_mode
        self._chunk_size = chunk_size
        self._tech_density = tech_density
        self._bio_density = bio_density
        self._use_val = use_val
        self._val_path = val_path
        self._val_iter = val_iter
        self._val_freq = val_freq
        self._val_transforms = val_transforms

        if self._val_transforms is None:
            self._val_transforms = []
            for transform in self._train_transforms:
                if not transform.augmentation:
                    self._val_transforms.append(transform)
        else:
            t_num = 0
            for transform in self._train_transforms:
                if not transform.augmentation:
                    t_num += 1
                    valid = False
                    for val_transform in self._val_transforms:
                        if val_transform.attributes == transform.attributes:
                            valid = True
                    if not valid:
                        raise ValueError("Validation transforms differ from training transforms")
            if len(self._val_transforms) != t_num:
                raise ValueError("Validation transforms differ from training transforms")

        self._use_cuda = use_cuda
        self._use_big = use_big
        self._random_seed = random_seed
        self._track_running_stats = track_running_stats
        self._max_step_size = max_step_size
        self._label_mappings = label_mappings
        self._hybrid_mode = hybrid_mode
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._splitting_redundancy = splitting_redundancy
        self._use_norm = use_norm
        self._class_weights = class_weights
        self._use_bias = use_bias
        self._norm_type = norm_type
        self._label_remove = label_remove
        self._sampling = sampling
        self._batch_avg = batch_avg
        self._kernel_size = kernel_size
        self._neighbor_nums = neighbor_nums
        self._dilations = dilations
        self._reductions = reductions
        self._first_layer = first_layer
        self._padding = padding
        self._centroids = centroids
        self._nn_center = nn_center
        self._optim_kernels = optim_kernels
        self._pl = pl
        self._dropout = dropout
        self._split_on_demand = split_on_demand
        self._split_jitter = split_jitter
        self._cp_norm = cp_norm
        self._architecture = architecture
        self._act = act
        self._epoch_size = epoch_size
        self._workers = workers
        self._voxel_sizes = voxel_sizes
        self._ssd_exclude = ssd_exclude
        self._ssd_include = ssd_include
        self._ssd_labels = ssd_labels
        self._exclude_borders = exclude_borders
        self._rebalance = rebalance

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
    def label_mappings(self):
        return self._label_mappings

    @property
    def hybrid_mode(self):
        return self._hybrid_mode

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def splitting_redundancy(self):
        return self._splitting_redundancy

    @property
    def use_norm(self):
        return self._use_norm

    @property
    def class_weights(self):
        return self._class_weights

    @property
    def use_bias(self):
        return self._use_bias

    @property
    def norm_type(self):
        return self._norm_type

    @property
    def label_remove(self):
        return self._label_remove

    @property
    def sampling(self):
        return self._sampling

    @property
    def batch_avg(self):
        return self._batch_avg

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def neighbor_nums(self):
        return self._neighbor_nums

    @property
    def dilations(self):
        return self._dilations

    @property
    def reductions(self):
        return self._reductions

    @property
    def first_layer(self):
        return self._first_layer

    @property
    def padding(self):
        return self._padding

    @property
    def centroids(self):
        return self._centroids

    @property
    def nn_center(self):
        return self._nn_center

    @property
    def optim_kernels(self):
        return self._optim_kernels

    @property
    def pl(self):
        return self._pl

    @property
    def dropout(self):
        return self._dropout

    @property
    def split_on_demand(self):
        return self._split_on_demand

    @property
    def split_jitter(self):
        return self._split_jitter

    @property
    def cp_norm(self):
        return self._cp_norm

    @property
    def architecture(self):
        return self._architecture

    @property
    def act(self):
        return self._act

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def workers(self):
        return self._workers

    @property
    def voxel_sizes(self):
        return self._voxel_sizes

    @property
    def ssd_exclude(self):
        return self._ssd_exclude

    @property
    def ssd_include(self):
        return self._ssd_include

    @property
    def ssd_labels(self):
        return self._ssd_labels

    @property
    def exclude_borders(self):
        return self._exclude_borders

    @property
    def rebalance(self):
        return self._rebalance

    @property
    def attr_dict(self):
        attr_dict = {'save_root': self._save_root,
                     'train_path': self._train_path,
                     'sample_num': self._sample_num,
                     'name': self._name,
                     'class_num': self._class_num,
                     'train_transforms': self._train_transforms,
                     'batch_size': self._batch_size,
                     'input_channels': self.input_channels,
                     'features': self._features,
                     'max_step_size': self._max_step_size,
                     'density_mode': self._density_mode,
                     'chunk_size': self._chunk_size,
                     'tech_density': self._tech_density,
                     'bio_density': self._bio_density,
                     'use_val': self._use_val,
                     'val_transforms': self._val_transforms,
                     'val_path': self._val_path,
                     'val_iter': self._val_iter,
                     'val_freq': self._val_freq,
                     'track_running_stats': self._track_running_stats,
                     'use_big': self._use_big,
                     'random_seed': self._random_seed,
                     'use_cuda': self._use_cuda,
                     'label_mappings': self._label_mappings,
                     'hybrid_mode': self._hybrid_mode,
                     'optimizer': self._optimizer,
                     'scheduler': self._scheduler,
                     'splitting_redundancy': self._splitting_redundancy,
                     'use_norm': self._use_norm,
                     'class_weights': self._class_weights,
                     'use_bias': self._use_bias,
                     'norm_type': self._norm_type,
                     'label_remove': self._label_remove,
                     'sampling': self._sampling,
                     'batch_avg': self._batch_avg,
                     'kernel_size': self._kernel_size,
                     'neighbor_nums': self._neighbor_nums,
                     'dilations': self._dilations,
                     'reductions': self._reductions,
                     'first_layer': self._first_layer,
                     'padding': self._padding,
                     'centroids': self._centroids,
                     'nn_center': self._nn_center,
                     'optim_kernels': self._optim_kernels,
                     'pl': self._pl,
                     'dropout': self._dropout,
                     'split_on_demand': self._split_on_demand,
                     'split_jitter': self._split_jitter,
                     'cp_norm': self._cp_norm,
                     'architecture': self._architecture,
                     'act': self._act,
                     'epoch_size': self._epoch_size,
                     'workers': self._workers,
                     'voxel_sizes': self._voxel_sizes,
                     'ssd_exclude': self._ssd_exclude,
                     'ssd_include': self._ssd_include,
                     'ssd_labels': self._ssd_labels,
                     'exclude_borders': self._exclude_borders,
                     'rebalance': self._rebalance}
        return attr_dict

    def save2pkl(self, path: str):
        attr_dict = self.attr_dict
        with open(path, 'wb') as f:
            pickle.dump(attr_dict, f)

    def load_from_pkl(self, path: str):
        """
        Load attribute dict from pickle file.

        Args:
            path: Path to pickle file which contains the attribute dictionary.
        """
        self.__init__(**basics.load_pkl(path))
        return self
