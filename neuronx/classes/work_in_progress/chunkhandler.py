# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import re
import glob
import pickle
import random
import time
import numpy as np
import open3d as o3d
from tqdm import tqdm
from multiprocessing import Process, Queue
from typing import Union, Tuple, List, Dict, Optional
from morphx.processing import clouds, objects
from morphx.preprocessing import splitting
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from sklearn.preprocessing import label_binarize
from syconn.reps.super_segmentation import SuperSegmentationDataset
from morphx.processing.objects import context_splitting_kdt


def worker_split(id_queue: Queue, chunk_queue: Queue, ssd: SuperSegmentationDataset, ctx: int,
                 base_node_dst: int, parts: Dict[str, List[int]], labels_itf: str,
                 label_mappings: List[Tuple[int, int]], split_jitter: int = 0):
    """
    Args:
        id_queue: Input queue with cell ids.
        chunk_queue: Output queue with cell chunks.
        ssd: SuperSegmentationDataset which contains the cells to which the chunkhandler should get applied.
        ctx: Context size for splitting.
        base_node_dst: Distance between base nodes. Corresponds to redundancy / number of chunks per cell.
        parts: Information about cell surface and organelles, Tuples like (voxel_param, feature) keyed by identifier
            compatible with syconn (e.g. 'sv' or 'mi').
        labels_itf: Label identifier for existing label predictions within the sso objects of the ssd dataset.
        label_mappings: Tuples where label at index 0 should get mapped to label at index 1.
        split_jitter: Derivation from context size during splitting.
    """
    while True:
        if not id_queue.empty():
            ssv_id = id_queue.get()
            sso = ssd.get_super_segmentation_object(ssv_id)
            vert_dc = {}
            label_dc = {}
            encoding = {}
            offset = 0
            obj_bounds = {}
            for ix, k in enumerate(parts):
                pcd = o3d.geometry.PointCloud()
                verts = sso.load_mesh(k)[1].reshape(-1, 3)
                pcd.points = o3d.utility.Vector3dVector(verts)
                pcd, idcs = pcd.voxel_down_sample_and_trace(parts[k][0], pcd.get_min_bound(), pcd.get_max_bound())
                idcs = np.max(idcs, axis=1)
                vert_dc[k] = np.asarray(pcd.points)
                obj_bounds[k] = [offset, offset + len(pcd.points)]
                offset += len(pcd.points)
                if k == 'sv':
                    # prepare mask for filtering background / unpredicted points
                    mask = None
                    if labels_itf == 'axoness':
                        # 0: dendrite, 1: axon, 2: soma, 3: bouton, 4: terminal, 5/6: background/unpredicted
                        labels_total = sso.label_dict()[labels_itf][idcs]
                        mask = labels_total < 5
                        labels_total = labels_total[mask]
                    elif labels_itf == 'spiness':
                        # 1: head, 0: neck, 2: shaft, 3: other, 4/5: background/unpredicted
                        labels_total = sso.label_dict()['axoness'][idcs]
                        spiness = sso.label_dict()['spiness'][idcs]
                        mask = np.logical_not(np.logical_or(labels_total > 4,
                                                            np.logical_and(labels_total == 0, spiness > 3)))
                        labels_total = labels_total[mask]
                        spiness = spiness[mask]
                        labels_total[labels_total != 0] = 3
                        labels_total[labels_total == 0] = spiness[labels_total == 0]
                    else:
                        labels_total = sso.label_dict()[labels_itf][idcs]
                        mask = np.ones(len(labels_total)).astype(bool)
                    labels = labels_total
                    vert_dc[k] = vert_dc[k][mask]
                else:
                    labels = np.ones(len(vert_dc[k])) + ix + labels_total.max()
                    encoding[k] = ix + 1 + labels_total.max()
                label_dc[k] = labels
            sample_feats = np.concatenate([[parts[k][1]] * len(vert_dc[k]) for k in parts]).reshape(-1, 1)
            sample_feats = label_binarize(sample_feats, classes=np.arange(len(parts)))
            sample_pts = np.concatenate([vert_dc[k] for k in parts])
            sample_labels = np.concatenate([label_dc[k] for k in parts])
            # mark cellular organelles to be excluded from loss calculation - see torchhandler for use of no_pred
            no_pred = list(encoding.keys())
            if not sso.load_skeleton():
                raise ValueError(f'Couldnt find skeleton of {sso}')
            nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
            hc = HybridCloud(nodes, edges, vertices=sample_pts, features=sample_feats, obj_bounds=obj_bounds,
                             no_pred=no_pred, labels=sample_labels, encoding=encoding)
            if label_mappings is not None:
                hc.map_labels(label_mappings)
            _ = hc.verts2node
            jitter = random.randint(0, split_jitter)
            node_arrs, source_nodes = splitting.split_single(hc, ctx + jitter, base_node_dst)
            for ix, node_arr in enumerate(node_arrs):
                sample, _ = objects.extract_cloud_subset(hc, node_arr)
                chunk_queue.put(sample)
        else:
            time.sleep(0.5)


class ChunkHandler:
    """ Helper class for loading, sampling and transforming chunks of different objects. Objects
        must be instances of a MorphX class. Uses :func:`morphx.preprocessing.splitting.split` to
        generate chunking information (or loads it if it is already available for the given chunk_size.

        :attr:`specific` defines two different modes. If the flag is False, the chunks are accessed
        rather randomly, not preserving their original position in the object. This mode can be
        used for training purposes. If :attr:`specific` is True, the chunks ca be requested from a
        specific index within a specific object. After processing the chunks, the results can
        then be mapped back to that position using :func:`map_predictions`. This mode can be used
        during inference when predictions of chunk are generated and should be inserted into the
        original object.
    """
    def __init__(self,
                 data: Union[str, SuperSegmentationDataset],
                 sample_num: int,
                 ctx_size: int = None,
                 transform: clouds.Compose = clouds.Compose([clouds.Identity()]),
                 specific: bool = False,
                 data_type: str = 'ce',
                 obj_feats: dict = None,
                 label_mappings: List[Tuple[int, int]] = None,
                 hybrid_mode: bool = False,
                 splitting_redundancy: int = 1,
                 label_remove: List[int] = None,
                 sampling: bool = True,
                 padding: int = None,
                 verbose: bool = False,
                 split_on_demand: bool = False,
                 split_jitter: int = 0,
                 epoch_size: int = None,
                 workers: int = 2,
                 voxel_sizes: Optional[dict] = None,
                 ssd_exclude: List[int] = None,
                 ssd_include: List[int] = None,
                 ssd_labels: str = None,
                 exclude_borders: int = 0,
                 rebalance: dict = None,
                 force_split: bool = True):
        """
        Args:
            data: Path to objects saved as pickle files. Existing chunking information would
                be available in the folder 'splitted' at this location.
            sample_num: Number of vertices which should be sampled from the surface of each chunk.
                Should be equal to the capacity of the given network architecture.
            transform: Transformations which should be applied to the chunks before returning them
                (e.g. see :func:`morphx.processing.clouds.Compose`)
            specific: Flag for setting mode of requesting specific or rather randomly drawn chunks.
            data_type: Type of dataset, 'ce': CloudEnsembles, 'hc': HybridClouds
            obj_feats: Only used when inputs are CloudEnsembles. Dict with feature array (1, n) keyed by
                the name of the corresponding object in the CloudEnsemble. The HybridCloud gets addressed
                with 'hc'.
            label_mappings: list of labels which should get replaced by other labels. E.g. [(1, 2), (3, 2)]
                means that the labels 1 and 3 will get replaced by 3.
            splitting_redundancy: indicates how many times each skeleton node is included in different contexts.
            label_remove: List of labels indicating which nodes should be removed from the dataset. This is
                is independent from the label_mappings, as the label removal is done during splitting.
            sampling: Flag for random sampling from the extracted subsets.
            padding: add padded points if a subset contains less points than there should be sampled.
            verbose: Return additional information about size of subsets.
            split_on_demand: Do not generate splitting information in advance, but rather generate chunks on the fly.
            split_jitter: Used only if split_on_demand = True. Adds jitter to the context size of the generated chunks.
            epoch_size: Parameter for epoch size that can be used when dataset size is unknown and epoch size should
                somehow be bounded.
            workers: Number of workers in case of ssd dataset.
            voxel_sizes: Voxelization options in case of ssd dataset use. Given as dict with voxel sizes keyed by
                cell part identifier (e.g. 'sv' or 'mi').
            exclude_borders: Offset radius (chunk_size - exclude_border) for excluding border regions of chunks from
                loss calculation.
            rebalance: dict for rebalancing of dataset if certain classes dominate. dict contains factor keyed by labels
                where the factor indicate how often the labels should get resampled.
            force_split: flag for resplitting of dataset.
        """
        if type(data) == SuperSegmentationDataset:
            self._data = data
        else:
            self._data = os.path.expanduser(data)
            if not os.path.exists(self._data):
                os.makedirs(self._data)
            if not split_on_demand:
                if not os.path.exists(self._data + 'splitted/'):
                    os.makedirs(self._data + 'splitted/')
                # Load chunks or split dataset into chunks if it was not done already
                self._splitfile = f'{self._data}splitted/s{ctx_size}_r{splitting_redundancy}_lr{label_remove}.pkl'
                self._splitted_objs = {}
                orig_splitfile = self._splitfile
                while os.path.exists(self._splitfile):
                    if not force_split:
                        with open(self._splitfile, 'rb') as f:
                            self._splitted_objs = pickle.load(f)
                        f.close()
                        break
                    else:
                        version = re.findall(r"v(\d+).", self._splitfile)
                        if len(version) == 0:
                            self._splitfile = self._splitfile[:-4] + '_v1.pkl'
                        else:
                            version = int(version[0])
                            self._splitfile = orig_splitfile[:-4] + f'_v{version+1}.pkl'
        self._voxel_sizes = dict(sv=80, mi=100, syn_ssv=100, vc=100)
        if voxel_sizes is not None:
            self._voxel_sizes = voxel_sizes
        self._sample_num = sample_num
        self._transform = transform
        self._specific = specific
        self._data_type = data_type
        self._obj_feats = obj_feats
        self._label_mappings = label_mappings
        self._hybrid_mode = hybrid_mode
        self._label_remove = label_remove
        self._sampling = sampling
        self._padding = padding
        self._verbose = verbose
        self._split_on_demand = split_on_demand
        self._ctx_size = ctx_size
        self._splitting_redundancy = splitting_redundancy
        self._split_jitter = split_jitter
        self._epoch_size = epoch_size
        self._workers = workers
        self._ssd_labels = ssd_labels
        self._ssd_exclude = ssd_exclude
        self._rebalance = rebalance
        self._exclude_borders = exclude_borders
        if ssd_exclude is None:
            self._ssd_exclude = []
        self._ssd_include = ssd_include
        if self._ssd_labels is None and type(self._data) == SuperSegmentationDataset:
            raise ValueError("ssd_labels must be specified when working with a SuperSegmentationDataset!")
        self._objs = {}
        self._chunk_list = []
        self._parts = {}

        # define loader function
        if type(data) == SuperSegmentationDataset:
            self._load_func = self.get_item_ssd
        else:
            self._load_func = self.get_item

        if type(self._data) == SuperSegmentationDataset:
            for key in self._obj_feats:
                self._parts[key] = [self._voxel_sizes[key], self._obj_feats[key]]
            # If ssd dataset is given, multiple workers are used for splitting the ssvs of the given dataset.
            self._obj_names = Queue()
            self._chunk_list = Queue(maxsize=10000)
            if self._ssd_include is None:
                print("Gathering sizes.")
                sizes = [sso.size for sso in self._data.ssvs]
                print("Done with sizes.")
                idcs = np.argsort(sizes)
                self._ssd_include = np.array(self._data.ssv_ids)[idcs[-200:]]
            for ssv in self._ssd_include:
                if ssv not in self._ssd_exclude:
                    self._obj_names.put(ssv)
            self._splitters = [Process(target=worker_split, args=(self._obj_names, self._chunk_list, self._data,
                                                                  self._ctx_size,
                                                                  self._ctx_size / self._splitting_redundancy,
                                                                  self._parts, self._ssd_labels,
                                                                  self._label_mappings, self._split_jitter))
                               for ix in range(workers)]
            for splitter in self._splitters:
                splitter.start()
        else:
            # load all available objects into memory
            files = glob.glob(data + '*.pkl')
            print("Splitting...")
            for file in tqdm(files):
                slashs = [pos for pos, char in enumerate(file) if char == '/']
                name = file[slashs[-1] + 1:-4]
                obj = self._adapt_obj(objects.load_obj(self._data_type, file))
                self._objs[name] = obj
                if name not in self._splitted_objs:
                    print(f"No splitting info found for {name}.")
                    self._splitted_objs[name] = \
                        splitting.split_single(obj, self._ctx_size, self._ctx_size / self._splitting_redundancy, 800)
                with open(self._splitfile, 'wb') as f:
                    pickle.dump(self._splitted_objs, f)
                f.close()
            if not self._specific:

                if split_on_demand:
                    for name in self._objs:
                        obj = self._objs[name]
                        base_nodes = np.arange(len(obj.nodes)).reshape(-1, 1)[obj.node_labels != -1]
                        base_nodes = np.random.choice(base_nodes, int(len(base_nodes) / 3), replace=True)
                        chunks = context_splitting_kdt(obj, base_nodes, self._ctx_size)
                        for chunk in chunks:
                            self._chunk_list.append((name, chunk))
                else:
                    for item in self._splitted_objs:
                        for idx in range(len(self._splitted_objs[item][0])):
                            self._chunk_list.append((item, idx))
                if self._rebalance is not None:
                    print("Rebalancing...")
                    balance = {}
                    for key in self._rebalance:
                        balance[key] = 0
                    for ix in tqdm(range(len(self._chunk_list))):
                        item = self._chunk_list[ix]
                        obj = self._objs[item[0]]
                        label = int(np.unique(obj.labels)[np.unique(obj.labels) < 4])
                        balance[label] += 1
                        for i in range(self._rebalance[label]):
                            self._chunk_list.append(item)
                            balance[label] += 1
                    print("Done with rebalancing!")
                    print(balance)
                random.shuffle(self._chunk_list)
        # Index of current chunk
        self._ix = 0
        self._size = len(self._chunk_list)

    def __len__(self):
        """ Depending on the mode either the sum of the number of chunks from each
            objects gets returned or the number of chunks of the last requested
            object in specific mode.
        """
        if self._epoch_size is not None:
            size = self._epoch_size
        elif self._ssd_include is not None:
            size = len(self._ssd_include)
        else:
            size = self._size
        return size

    def __getitem__(self, item: Union[int, Tuple[str, int]]):
        """ Returns either a chunk from a specific location and object or iterates
            the objects sequentially and returns the chunks in the order of the
            chunking information. If the sampled PointCloud contains no vertices, a
            PointCloud with `self._sample_num` vertices and labels is returned, where
            all numbers are set to 0.

        Args:
            item: With :attr:`specific` as False, this parameter is a simple integer indexing
                the training examples. With true it must be a tuple of the filename of the
                requested object and the index of the chunk within that object. E.g. if chunk
                5 from object in pickle file object.pkl is requested, this would be ('object', 5).
        """
        vert_num = None
        sample, ixs, source_node = self._load_func(item)
        if sample is None and self._specific:
            return None, None
        if self._verbose:
            vert_num = len(sample.vertices)
        if self._sampling:
            sample, ixs = clouds.sample_cloud(sample, self._sample_num, padding=self._padding)
        if self._exclude_borders != 0:
            max_label = sample.labels.max()
            sample.mark_borders(max_label + 1, self._ctx_size - self._exclude_borders, centroid=source_node)
            sample.encoding['border'] = max_label + 1
            sample.no_pred.append('border')
        # Apply transformations (e.g. Composition of Rotation and Normalization)
        self._transform(sample)
        if len(sample.vertices) > 0:
            if self._verbose:
                # Return sample, indices from where sample points were taken and number of vertices in the subset
                return sample, ixs, vert_num
            elif self._specific:
                # Return sample and indices from where sample points were taken
                return sample, ixs
            else:
                # Return only sample in non-specific mode
                return sample
        else:
            if self._verbose:
                return None, np.empty(0), 0
            elif self._specific:
                return None, np.empty(0)
            else:
                return None

    def get_item_ssd(self, item: Union[int, Tuple[str, int]]):
        """
        Loading method used when data is given in form of ssd dataset. Uses multiple workers for splitting of the ssvs.
        """
        while self._chunk_list.empty():
            if self._obj_names.empty():
                np.random.shuffle(self._ssd_include)
                for ssv in self._ssd_include:
                    if ssv not in self._ssd_exclude:
                        self._obj_names.put(ssv)
            time.sleep(0.5)
        return self._chunk_list.get(), None, None

    def get_item(self, item: Union[int, Tuple[str, int]]):
        """
        Loading method used when data is given as MorphX objects.
        """
        if self._split_on_demand and not self._specific:
            if item == self._size-1:
                self._chunk_list = []
                for name in self._objs:
                    obj = self._objs[name]
                    jitter = random.randint(-self._split_jitter, self._split_jitter)
                    base_nodes = np.arange(len(obj.nodes)).reshape(-1, 1)[obj.node_labels != -1]
                    base_nodes = np.random.choice(base_nodes, int(len(base_nodes) / 3), replace=True)
                    node_arrs = context_splitting_kdt(obj, base_nodes, self._ctx_size + jitter)
                    for node_arr in node_arrs:
                        self._chunk_list.append((name, node_arr))
            name, node_arr = self._chunk_list[item]
            obj = self._objs[name]
            sample, idcs = objects.extract_cloud_subset(obj, node_arr)
            return sample, idcs, None
        if type(item) != tuple:
            item = self._chunk_list[item]
        # self._splitted_objs: name -> (node_arrs, source_nodes)
        obj = self._objs[item[0]]
        node_arrs, source_nodes = self._splitted_objs[item[0]]
        next_ix = item[1]
        if next_ix >= len(node_arrs) or abs(next_ix) > len(node_arrs):
            if self._specific:
                return None, None, None
            next_ix = item[1] % len(node_arrs)
        node_arr = node_arrs[next_ix]
        source_node = source_nodes[next_ix]
        sample, idcs = objects.extract_cloud_subset(obj, node_arr)
        return sample, idcs, obj.nodes[source_node]

    def terminate(self):
        if type(self._data) == SuperSegmentationDataset:
            for splitter in self._splitters:
                splitter.terminate()
                splitter.join()
                splitter.close()

    @property
    def obj_names(self):
        return self._objs.keys()

    @property
    def splitfile(self):
        return self._splitfile

    @property
    def sample_num(self):
        return self._sample_num

    def switch_mode(self):
        """ Switch specific mode on and off. """
        self._specific = not self._specific

    def get_obj_length(self, name: str):
        """ Returns the number of chunks for a specific object.

        Args:
            name: Filename of the requested object. If the file is object.pkl this would be 'object'.
        """
        return len(self._splitted_objs[name][0])

    def get_obj_info(self, name: str):
        obj = self._objs[name]
        attr_dict = {'vertex_num': len(obj.vertices), 'node_num': len(obj.nodes),
                     'types': list(np.unique(obj.types, return_counts=True)),
                     'labels': list(np.unique(obj.labels, return_counts=True)), 'length': self.get_obj_length(name)}
        return attr_dict

    def get_set_info(self):
        """ Returns a dict with information about the specified dataset. """
        total_attr_dict = {'vertex_num': 0, 'node_num': 0, 'types': [np.array([]), np.array([])],
                           'labels': [np.array([]), np.array([])], 'length': 0}
        for name in self.obj_names:
            attr_dict = self.get_obj_info(name)
            total_attr_dict[name] = attr_dict
            total_attr_dict['vertex_num'] += attr_dict['vertex_num']
            total_attr_dict['node_num'] += attr_dict['node_num']
            total_attr_dict['length'] += attr_dict['length']
            for key in ['labels', 'types']:
                labels = attr_dict[key]
                total_labels = total_attr_dict[key]
                for source_ix, label in enumerate(labels[0]):
                    if label in total_labels[0]:
                        # add label counts of current obj to total
                        target_ix = int(np.argwhere(total_labels[0] == label))
                        total_labels[1][target_ix] += labels[1][source_ix]
                    else:
                        # append label and label counts
                        total_labels[0] = np.append(total_labels[0], int(label))
                        total_labels[1] = np.append(total_labels[1], int(labels[1][source_ix]))
                assert total_attr_dict['labels'][1].sum() == total_attr_dict['vertex_num']
        return total_attr_dict

    def _adapt_obj(self, obj: Union[CloudEnsemble, HybridCloud]) -> Union[CloudEnsemble, HybridCloud]:
        """ Adds given parameters like features or label mappings to the loaded object. """
        # transform to HybridCloud:
        if self._hybrid_mode and isinstance(obj, CloudEnsemble):
            obj = obj.hc
        # change features
        if self._obj_feats is not None:
            if isinstance(obj, CloudEnsemble):
                for name in self._obj_feats:
                    feat_line = self._obj_feats[name]
                    if name == 'sv':
                        name = 'hc'
                    subcloud = obj.get_cloud(name)
                    if subcloud is not None:
                        if isinstance(feat_line, dict):
                            subcloud.types2feat(feat_line)
                        else:
                            if type(feat_line) == int:
                                feats = np.ones((len(subcloud.vertices), 1)) * feat_line
                                if len(feats) != 0:
                                    feats = label_binarize(feats, classes=np.arange(len(self._obj_feats)))
                            else:
                                feats = np.ones((len(subcloud.vertices), len(feat_line)))
                                feats[:] = feat_line
                            subcloud.set_features(feats)
            elif self._hybrid_mode:
                feats = np.ones(len(obj.vertices)).reshape(-1, 1)
                feats[:] = self._obj_feats['hc']
                obj.set_features(feats)
        # remove nodes of given labels
        if self._label_remove is not None:
            obj.remove_nodes(self._label_remove)
        # change labels
        if self._label_mappings is not None:
            obj.map_labels(self._label_mappings)
        return obj
