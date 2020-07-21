import os
import pickle
from morphx.data.chunkhandler import ChunkHandler
from morphx.processing import clouds
import numpy as np


def produce_chunks():
    """ Create and save all resulting chunks of an dataset. """
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    chunk_size = 10000
    identity = clouds.Compose([clouds.Identity()])
    center = clouds.Compose([clouds.Center()])
    path = os.path.expanduser('~/thesis/gt/cmn/dnh/voxeled/')
    save_path = f'{path}examples/'
    ch = ChunkHandler(path, sample_num=5000, density_mode=False, tech_density=100, bio_density=100, specific=True,
                      chunk_size=chunk_size, obj_feats=features, transform=identity, splitting_redundancy=2,
                      label_mappings=[(5, 3), (6, 4)], label_remove=None, sampling=True, verbose=True)
    ch_transform = ChunkHandler(path, sample_num=5000, density_mode=False, tech_density=100, bio_density=100, specific=True,
                                chunk_size=chunk_size, obj_feats=features, transform=center, splitting_redundancy=2,
                                label_mappings=[(5, 3), (6, 4)], label_remove=None, sampling=True, verbose=True)
    vert_nums = []
    counter = 0
    chunk_num = 0
    total = None
    for item in ch.obj_names:
        total_cell = None
        chunk_num += ch.get_obj_length(item)
        for i in range(ch.get_obj_length(item)):
            sample, idcs, vert_num = ch[(item, i)]
            sample_t, _, _ = ch_transform[(item, i)]
            vert_nums.append(vert_num)
            if not os.path.exists(save_path + f'{item}/'):
                os.makedirs(save_path + f'{item}/')
            if vert_num < ch.sample_num:
                counter += 1
            with open(f'{save_path}{item}/{i}.pkl', 'wb') as f:
                pickle.dump([sample, sample_t], f)
            if total_cell is None:
                total_cell = sample
            else:
                total_cell = clouds.merge_clouds([total_cell, sample])
        if total is None:
            total = total_cell
        else:
            total = clouds.merge_clouds([total, total_cell])
        with open(f'{save_path}{item}/total.pkl', 'wb') as f:
            pickle.dump(total_cell, f)
    with open(f'{save_path}total.pkl', 'wb') as f:
        pickle.dump(total, f)
    vert_nums = np.array(vert_nums)
    print(f"Min: {vert_nums.min()}")
    print(f"Max: {vert_nums.max()}")
    print(f"Mean: {vert_nums.mean()}")
    print(f"Chunks with less points than requested: {counter}/{chunk_num}")
    with open(f'{save_path}{chunk_size}_vertnums.pkl', 'wb') as f:
        pickle.dump(vert_nums, f)
    f.close()


def compare_chunks():
    """ Create chunks with different ChunkHandlers and compare the results. """
    path = os.path.expanduser('~/thesis/gt/test_gt/')
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    transforms1 = clouds.Compose([clouds.Center(), clouds.RandomShear(limits=(-0.5, 0.5))])
    ch1 = ChunkHandler(path, sample_num=10000, density_mode=False, specific=True, chunk_size=10000,
                       obj_feats=features, transform=transforms1)
    transforms2 = clouds.Compose([clouds.Center()])
    ch2 = ChunkHandler(path, sample_num=10000, density_mode=False, specific=True, chunk_size=10000,
                       obj_feats=features, transform=transforms2)
    item = ch1.obj_names[2]
    for i in range(4):
        sample1, _ = ch1[(item, i)]
        sample2, _ = ch2[(item, i)]
        samples = [sample1, sample2]
        with open(f'{path}shear_vis/{item}_{i}_05.pkl', 'wb') as f:
            pickle.dump(samples, f)
        f.close()


def apply_chunkhandler():
    path = os.path.expanduser('~/thesis/gt/cmn/dnh/test/')
    chunk_size = 10000
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    identity = clouds.Compose([clouds.Identity()])
    ch = ChunkHandler(path, sample_num=5000, density_mode=False, tech_density=100, bio_density=100, specific=False,
                      chunk_size=chunk_size, obj_feats=features, transform=identity, splitting_redundancy=1,
                      sampling=True, split_on_demand=True, split_jitter=10000)
    for r in range(15):
        for ix in range(len(ch)):
            sample = ch[ix]


if __name__ == '__main__':
    apply_chunkhandler()
