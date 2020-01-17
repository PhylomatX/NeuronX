import os
import numpy as np
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud
from morphx.data.chunkhandler import ChunkHandler
from morphx.postprocessing.mapping import PredictionMapper


def prediction_sanity():
    data_path = os.path.expanduser('~/thesis/gt/gt_poisson/single/')
    save_path = os.path.expanduser('~/thesis/gt/gt_poisson/single/chunks/')
    radius = 20000
    npoints = 5000
    cl = ChunkHandler(data_path, radius, npoints)
    pm = PredictionMapper(data_path, data_path + 'predicted/', radius)

    merged = None
    cl.set_specific_mode(True)
    size = cl.get_hybrid_length('sso_46319619_c')
    for idx in range(size):
        sample = cl[('sso_46319619_c', idx)]
        labels = np.ones(len(sample.labels))*idx
        pred_cloud = PointCloud(sample.vertices, labels)
        if merged is None:
            merged = pred_cloud
        else:
            merged = clouds.merge_clouds(merged, pred_cloud)
        clouds.save_cloud(pred_cloud, save_path, 'chunk_{}'.format(idx))
        pm.map_predictions(pred_cloud, 'sso_46319619_c', idx)
    pm.save_prediction()
    clouds.save_cloud(merged, save_path, 'merged')


if __name__ == '__main__':
    prediction_sanity()
