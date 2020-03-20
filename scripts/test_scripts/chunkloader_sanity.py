import os
import time
import ipdb
import numpy as np
from tqdm import tqdm
from morphx.data.chunkhandler import ChunkHandler
from morphx.processing import clouds
from morphx.data import basics

if __name__ == '__main__':
    tech_density = 1500
    bio_density = 20
    sample_num = 28000
    data_path = os.path.expanduser('~/thesis/gt/20_02_20/poisson_verts2node/')

    chunk_size = 30000

    ch = ChunkHandler(data_path, sample_num, density_mode=False, chunk_size=chunk_size, bio_density=bio_density,
                      tech_density=tech_density, specific=False,
                      obj_feats={'hc': np.array([1, 0, 0, 0]), 'mi': np.array([0, 1, 0, 0]),
                                 'vc': np.array([0, 0, 1, 0]), 'sy': np.array([0, 0, 0, 1])},
                      label_mappings=[(4, 2), (5, 2), (6, 1)])
    print("Start...")
    times = []
    point_nums = []
    for i in tqdm(range(len(ch))):
        start = time.time()
        res = ch[i]
        if res is not None:
            chunk = res[0]
            point_num = res[1]
        else:
            continue
        point_nums.append(point_num)
        times.append((time.time() - start))
    point_nums = np.array(point_nums)
    ipdb.set_trace()

    # for obj in ch.obj_names:
    #     for ix in tqdm(range(ch.get_obj_length(obj))):
    #         start = time.time()
    #         chunk, idcs = ch[(obj, ix)]
    #         times.append(time.time() - start)

    # (array([12965, 2232, 590, 184, 67, 25, 5, 5, 1,
    #         1]), array([1.00000e+00, 2.62930e+04, 5.25850e+04, 7.88770e+04, 1.05169e+05,
    #                     1.31461e+05, 1.57753e+05, 1.84045e+05, 2.10337e+05, 2.36629e+05,
    #                     2.62921e+05]))

    summed = np.array(times).sum()
    print(summed / len(times))

