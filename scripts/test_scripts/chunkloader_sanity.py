import os
import time
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

    ch = ChunkHandler(data_path, sample_num, density_mode=True, bio_density=bio_density, tech_density=tech_density,
                      specific=False)
    print("Start...")
    times = []

    for i in range(len(ch)):
        start = time.time()
        chunk = ch[i]
        times.append((time.time() - start))

    # for obj in ch.obj_names:
    #     for ix in tqdm(range(ch.get_obj_length(obj))):
    #         start = time.time()
    #         chunk, idcs = ch[(obj, ix)]
    #         times.append(time.time() - start)

    summed = np.array(times).sum()
    print(summed / len(times))

