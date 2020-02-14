import os
from tqdm import tqdm
from morphx.data.chunkhandler import ChunkHandler
from morphx.processing import clouds, objects

if __name__ == '__main__':
    context = 15000
    points = 10000
    data_path = os.path.expanduser('~/thesis/gt/gt_ensembles/ads/')
    save_path = os.path.expanduser('~/thesis/gt/gt_ensembles/ads/chunks/' + f'{context}_{points}/')
    ch = ChunkHandler(data_path, context, points, specific=True)

    # sample, idcs = ch[('sso_491527', 0)]
    # sample.save2pkl(save_path, name='sso_491527_0')

    for obj in ch.obj_names:
        full = None
        samples = []
        total = None
        for ix in tqdm(range(ch.get_obj_length(obj))):
            chunk, idcs = ch[(obj, ix)]
            samples.append(chunk)
            if full is None:
                full = chunk
            else:
                full = clouds.merge_clouds([full, chunk])

        full.save2pkl(save_path, name=obj)
        objects.save2pkl(samples, save_path, name=obj+'_samples')
