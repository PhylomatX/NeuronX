import os
import pickle
from neuronx.classes.chunkhandler import ChunkHandler
from neuronx.classes.torchhandler import TorchHandler
from morphx.processing import clouds
import numpy as np
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset


def produce_chunks(chunk_size: int, sample_num: int):
    """ Create and analyse all resulting chunks of an dataset. """
    features = {'hc': np.array([1])}
    center = clouds.Compose([clouds.Identity()])
    path = os.path.expanduser('~/working_dir/gt/cmn/dnh/voxeled/')
    save_path = f'{path}analysis/'
    ch = ChunkHandler(path, sample_num=sample_num, density_mode=False, ctx_size=chunk_size, obj_feats=features,
                      transform=center, splitting_redundancy=5, label_mappings=[(2, 0), (5, 1), (6, 2)],
                      label_remove=[2], sampling=True, verbose=True, specific=True, hybrid_mode=True)
    vert_nums = []
    counter = 0
    chunk_num = 0
    for item in ch.obj_names:
        chunk_num += ch.get_obj_length(item)
        for i in range(ch.get_obj_length(item)):
            sample, idcs, vert_num = ch[(item, i)]
            if vert_num < 10000:
                if not os.path.exists(save_path + f'examples/{item}/'):
                    os.makedirs(save_path + f'examples/{item}/')
                with open(f'{save_path}examples/{item}/{i}.pkl', 'wb') as f:
                    pickle.dump([sample, sample], f)
            vert_nums.append(vert_num)
            if vert_num < ch.sample_num:
                counter += 1
    vert_nums = np.array(vert_nums)
    analysis = f"Min: {vert_nums.min()}\nMax: {vert_nums.max()}\nMean: {vert_nums.mean()}\nChunks with less points than requested: {counter}/{chunk_num}"
    print(analysis)
    with open(f'{save_path}{chunk_size}_vertnums.pkl', 'wb') as f:
        pickle.dump(vert_nums, f)
    with open(f'{save_path}{chunk_size}_{sample_num}.txt', 'w') as f:
        f.write(analysis)
    f.close()
    return counter / chunk_num


def compare_transforms(chunk_size: int, sample_num: int):
    """ Create and save all resulting chunks of an dataset with different transforms """
    # features = {'hc': np.array([1, 0, 0, 0]),
    #             'mi': np.array([0, 1, 0, 0]),
    #             'vc': np.array([0, 0, 1, 0]),
    #             'sy': np.array([0, 0, 0, 1])}
    features = {'hc': np.array([1])}
    identity = clouds.Compose([clouds.Identity()])
    center = clouds.Compose([clouds.Center()])
    path = os.path.expanduser('~/thesis/gt/cmn/dnh/voxeled/')
    save_path = f'{path}examples/'
    ch = ChunkHandler(path, sample_num=sample_num, density_mode=False, tech_density=100, bio_density=100, specific=True,
                      ctx_size=chunk_size, obj_feats=features, transform=identity, splitting_redundancy=2,
                      label_mappings=[(5, 3), (6, 4)], label_remove=None, sampling=True, verbose=True)
    ch_transform = ChunkHandler(path, sample_num=5000, density_mode=False, tech_density=100, bio_density=100, specific=True,
                                ctx_size=chunk_size, obj_feats=features, transform=center, splitting_redundancy=2,
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
    path = os.path.expanduser('~/thesis/current_work/augmentation_tests/')
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    transforms1 = clouds.Compose([clouds.Center(), clouds.RandomScale(distr_scale=0.6, distr='uniform')])
    ch1 = ChunkHandler(path, sample_num=4000, density_mode=False, specific=True, ctx_size=4000,
                       obj_feats=features, transform=transforms1)
    transforms2 = clouds.Compose([clouds.Center()])
    ch2 = ChunkHandler(path, sample_num=4000, density_mode=False, specific=True, ctx_size=4000,
                       obj_feats=features, transform=transforms2)
    save_path = path+'scale/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for item in ch1.obj_names:
        for i in range(10):
            sample1, _ = ch1[(item, i)]
            sample2, _ = ch2[(item, i)]
            samples = [sample1, sample2]
            # meshes = [clouds.merge_clouds([sample1, meshes[0]]), clouds.merge_clouds([sample2, meshes[0]])]
            with open(f'{save_path}{item}_{i}.pkl', 'wb') as f:
                pickle.dump(samples, f)
            f.close()
            # with open(f'{save_path}{item}_{i}_meshes.pkl', 'wb') as f:
            #     pickle.dump(meshes, f)
            # f.close()


def apply_chunkhandler(save_path: str):
    path = os.path.expanduser('~/thesis/gt/20_09_27/voxeled/test/')
    chunk_size = 12000
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    identity = clouds.Compose([clouds.Center()])
    ch = ChunkHandler(path, sample_num=10000, density_mode=False, specific=False, ctx_size=chunk_size, obj_feats=features,
                      transform=identity, splitting_redundancy=1, sampling=True, split_on_demand=False, label_remove=[-2])
    info = ch.get_set_info()
    print(info['node_labels'])
    print(info['labels'])
    import ipdb
    ipdb.set_trace()
    # for ix in range(0, len(ch), 2):
    #     print(ix)
    #     sample1 = ch[ix]
    #     sample2 = ch[ix+1]
        # sample.mark_borders(10)
        # verts_centroid = np.array(sample.vertices)
        # centroid = np.mean(sample.vertices, axis=0)
        # verts_centroid = np.concatenate((verts_centroid, centroid.reshape((-1, 3))))
        # labels_centroid = np.concatenate((sample.labels, np.array(20).reshape(-1, 1)))
        # sample = PointCloud(vertices=verts_centroid, labels=labels_centroid)


def apply_torchhandler():
    path = os.path.expanduser('~/thesis/gt/cmn/dnh/test/')
    chunk_size = 5000
    features = {'hc': np.array([1])}
    identity = clouds.Compose([clouds.Center()])
    th = TorchHandler(path, sample_num=5000, density_mode=False, tech_density=100, bio_density=100, specific=False,
                      ctx_size=chunk_size, obj_feats=features, transform=identity, splitting_redundancy=1,
                      sampling=True, split_on_demand=True, nclasses=4, feat_dim=1, hybrid_mode=True,
                      exclude_borders=True)
    for ix in range(len(th)):
        sample = th[ix]


def apply_chunkhandler_ssd():
    data = SuperSegmentationDataset(working_dir="/wholebrain/songbird/j0126/areaxfs_v6/")
    ssd_include = [491527, 1090051]
    chunk_size = 4000
    features = {'sv': 1, 'mi': 2, 'vc': 3, 'syn_ssv': 4}
    transform = clouds.Compose([clouds.Center()])

    ch = ChunkHandler(data=data, sample_num=4000, density_mode=False, specific=False, ctx_size=chunk_size,
                      obj_feats=features, splitting_redundancy=1, sampling=True,
                      transform=transform, ssd_include=ssd_include, ssd_labels='axoness',
                      label_mappings=[(3, 2), (4, 3), (5, 1), (6, 1)])

    save_path = os.path.expanduser('~/thesis/current_work/chunkhandler_tests/')
    ix = 0
    while ix < 500:
        sample1 = ch[ix]
        sample2 = ch[ix+1]
        ix += 2
        sample = [sample1, sample2]
        with open(f'{save_path}{ix}.pkl', 'wb') as f:
            pickle.dump(sample, f)
        f.close()
    ch.terminate()


if __name__ == '__main__':
    report = 'context, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536\n'
    for context in [1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000]:
        context_report = f'{context / 1000}'
        for point_num in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            print(f'Analysing context {context}, point_num {point_num}')
            percentage = round(produce_chunks(context, point_num), 2)
            print(percentage)
            context_report += f', {percentage}'
        context_report += '\n'
        report += context_report
    print(report)
    path = os.path.expanduser('~/working_dir/gt/cmn/dnh/voxeled/')
    save_path = f'{path}analysis/'
    with open(f'{save_path}report.txt', 'w') as f:
        f.write(report)
    f.close()
