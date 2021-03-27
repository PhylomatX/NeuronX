import os
import re
import glob
import csv
from tqdm import tqdm
import numpy as np
from morphx.processing import ensembles
from syconn.reps.super_segmentation import SuperSegmentationDataset


def get_sso_specs(set_path: str, out_path: str, ssd: SuperSegmentationDataset):
    set_path = os.path.expanduser(set_path)
    out_path = os.path.expanduser(out_path)
    files = glob.glob(set_path + '*.pkl')
    total_edge_length = 0
    total_voxel_size = 0
    for file in tqdm(files):
        sso_id = int(re.findall(r"/sso_(\d+).", file)[0])
        sso = ssd.get_super_segmentation_object(sso_id)
        total_edge_length += sso.total_edge_length()
        total_voxel_size += sso.size
        info = f'{sso_id}:\nskeleton path length:\t{sso.total_edge_length()}\nvoxel size:\t{sso.size}\n\n'
        with open(out_path, 'a') as f:
            f.write(info)
        f.close()
    with open(out_path, 'a') as f:
        f.write(f'total edge length: {total_edge_length}\ntotal voxel size: {total_voxel_size}')
    f.close()


def dataspecs2csv(set_paths: dict, out_path: str, ssds: dict):
    out_path = os.path.expanduser(out_path)
    out_file = open(out_path, 'w')
    spec_writer = csv.writer(out_file, delimiter=',')
    for key in set_paths:
        ssd = ssds[key]
        spec_writer.writerow([key, 'edge_length', 'size', 'n_dendrite', 'n_axon', 'n_soma', 'n_bouton', 'n_terminal',
                              'n_neck', 'n_head', 'v_dendrite', 'v_axon', 'v_soma', 'v_bouton', 'v_terminal', 'v_neck',
                              'v_head'])
        set_path = os.path.expanduser(set_paths[key])
        files = glob.glob(set_path + '*.pkl')
        for file in tqdm(files):
            sso_id = int(re.findall(r"/sso_(\d+).", file)[0])
            sso = ssd.get_super_segmentation_object(sso_id)
            ce = ensembles.ensemble_from_pkl(file)
            n_unique = np.unique(ce.node_labels, return_counts=True)
            v_unique = np.unique(ce.labels, return_counts=True)
            n_list = []
            n_dict = {}
            for ix, n in enumerate(n_unique[0]):
                n_dict[n] = n_unique[1][ix]
            v_list = []
            v_dict = {}
            for ix, v in enumerate(v_unique[0]):
                v_dict[v] = v_unique[1][ix]
            for i in range(7):
                try:
                    n_list.append(n_dict[i])
                except KeyError:
                    n_list.append(0)
                try:
                    v_list.append(v_dict[i])
                except KeyError:
                    v_list.append(0)
            spec_writer.writerow([sso_id, int(sso.total_edge_length()), sso.size, *n_list, *v_list])
        out_file.write('\n\n\n\n')
    out_file.close()


if __name__ == '__main__':
    # paths = dict(TRAIN='~/working_dir/gt/cmn/dnh/voxeled/',
    #              TEST='~/working_dir/gt/cmn/dnh/voxeled/evaluation/')

    paths = dict(TEST='~/working_dir/gt/cmn/ads/train/voxeled/')

    # ssds = dict(TRAIN=SuperSegmentationDataset("/wholebrain/scratch/areaxfs3/"),
    #             TEST=SuperSegmentationDataset("/wholebrain/songbird/j0126/areaxfs_v6/"))

    ssds = dict(TEST=SuperSegmentationDataset("/wholebrain/scratch/areaxfs3/"))

    dataspecs2csv(paths, '~/working_dir/gt/cmn/ads/test.csv', ssds)

    # get_sso_specs('~/thesis/gt/20_09_27/voxeled/train/', '~/thesis/gt/20_09_27/voxeled/train_info.txt',
    #               ssd=SuperSegmentationDataset("/wholebrain/songbird/j0126/areaxfs_v6/"))

