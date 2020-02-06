import os
import math
import ipdb
import numpy as np
from tqdm import tqdm
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud
from morphx.data.chunkhandler import ChunkHandler
from morphx.postprocessing.mapping import PredictionMapper


def prediction_sanity():
    data_path = os.path.expanduser('~/thesis/gt/gt_poisson/single/')
    save_path = os.path.expanduser('~/thesis/gt/gt_poisson/single/chunks/')
    radius = 20000
    npoints = 5000

    cl = ChunkHandler(data_path, radius, npoints, specific=True)
    pm = PredictionMapper(data_path, data_path + 'predicted/', radius)

    merged = None
    size = cl.get_hybrid_length('sso_46319619_c')
    for idx in range(size):
        sample, centroid = cl[('sso_46319619_c', idx)]
        labels = np.ones(len(sample.labels))*idx
        pred_cloud = PointCloud(sample.vertices, labels)
        if merged is None:
            merged = pred_cloud
        else:
            merged = clouds.merge_clouds([merged, pred_cloud])
        clouds.save_cloud(pred_cloud, save_path, 'chunk_{}'.format(idx))
        pm.map_predictions(pred_cloud, 'sso_46319619_c', idx)
    pm.save_prediction()
    clouds.save_cloud(merged, save_path, 'merged')


def batch_prediction():
    data_path = os.path.expanduser('~/thesis/gt/gt_poisson/single/')
    save_path = os.path.expanduser('~/thesis/gt/gt_poisson/single/predicted/')
    radius = 10000
    npoints = 5000

    transforms = clouds.Compose([clouds.Normalization(radius), clouds.Center()])

    ch = ChunkHandler(data_path, radius, npoints, transform=transforms, specific=True)
    pm = PredictionMapper(data_path, save_path, radius)

    merged = None
    batch_size = 16
    hc = 'sso_46319619_c'
    print(ch.get_hybrid_length(hc))
    for batch in tqdm(range(math.ceil(ch.get_hybrid_length(hc) / batch_size))):
        t_pts = np.zeros((batch_size, npoints, 3))
        centroids = []

        for j in range(batch_size):
            chunk, centroid = ch[(hc, batch * batch_size + j)]
            centroids.append(centroid)
            t_pts[j] = chunk.vertices

        for j in range(batch_size):
            if not np.all(t_pts[j] == 0):
                # generate dummy output
                output = np.ones(npoints)*(batch * batch_size + j)
                prediction = PointCloud(t_pts[j], output)

                # Apply invers transformations
                prediction.move(centroids[j])
                prediction.scale(radius)

                if merged is None:
                    merged = prediction
                else:
                    merged = clouds.merge_clouds([merged, prediction])

                pm.map_predictions(prediction, hc, batch * batch_size + j)
            else:
                print('Zeros detected.')
    pm.save_prediction()
    clouds.save_cloud(merged, save_path, 'merged')


if __name__ == '__main__':
    batch_prediction()
