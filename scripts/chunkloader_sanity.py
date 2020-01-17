import os
import pickle
import numpy as np
from morphx.data.chunkhandler import ChunkHandler
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud

if __name__ == '__main__':
    data_path = os.path.expanduser('~/thesis/gt/gt_poisson/single/')
    cl = ChunkHandler(data_path, 20000, 800)
    full = None
    for idx in range(len(cl)):
        sample = cl[idx]
        labels = np.ones((len(sample.vertices), 1)) * idx
        sample = PointCloud(sample.vertices, labels=labels)
        if full is None:
            full = sample
        else:
            full = clouds.merge_clouds(full, sample)

    with open(data_path + 'chunks.pkl', 'wb') as f:
        pickle.dump(full, f)
    f.close()
