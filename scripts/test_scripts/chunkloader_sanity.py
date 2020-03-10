import os
from tqdm import tqdm
from morphx.data.chunkhandler import ChunkHandler
from morphx.processing import clouds
from morphx.data import basics

if __name__ == '__main__':
    tech_density = 1500
    bio_density = 100
    sample_num = 28000
    data_path = os.path.expanduser('~/thesis/gt/20_02_20/poisson/')
    save_path = os.path.expanduser(f'~/thesis/gt/20_02_20/poisson/samples/d{bio_density}/')

    ch = ChunkHandler(data_path, sample_num, density_mode=True, bio_density=bio_density, tech_density=tech_density,
                      specific=True)

    for obj in ch.obj_names:
        full = None
        samples = []
        for ix in tqdm(range(ch.get_obj_length(obj))):
            chunk, idcs, bfs = ch[(obj, ix)]
            samples.append([chunk, bfs])
            if full is None:
                full = chunk
            else:
                full = clouds.merge_clouds([full, chunk])
        full.save2pkl(f'{save_path}{obj}_d{bio_density}.pkl')
        basics.save2pkl(samples, save_path, name=f'{obj}_d{bio_density}_samples')
