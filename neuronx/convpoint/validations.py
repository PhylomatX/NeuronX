import os
import math
import torch
import time
import random
import numpy as np
from tqdm import tqdm
from morphx.processing import clouds
from morphx.data.chunkhandler import ChunkHandler
from morphx.classes.pointcloud import PointCloud
from morphx.postprocessing.mapping import PredictionMapper
from elektronn3.models.convpoint import SegSmall, SegBig


def validation_thread(args):
    save_root = os.path.expanduser(args[0])
    data_path = os.path.expanduser(args[1])
    radius = args[2]
    npoints = args[3]
    name = args[4]
    nclasses = args[5]
    transforms = args[6]
    batch_size = args[7]
    use_cuda = args[8]
    input_channels = args[9]
    use_big = args[10]
    random_seed = args[11]

    # set random seeds to ensure compareability of different trainings
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if use_big:
        model = SegBig(input_channels, nclasses)
    else:
        model = SegSmall(input_channels, nclasses)

    full = torch.load(save_root + '/' + name + '/state_dict.pth')
    model.load_state_dict(full['model_state_dict'])
    model.to(device)
    model.eval()

    # load scripted model
    # model_path = save_root + '/' + name + '/model.pts'
    # model = torch.jit.load(model_path, map_location=device)
    # model.eval()

    model_time = 0
    model_counter = 0

    transforms = clouds.Compose(transforms)
    ch = ChunkHandler(data_path, radius, npoints, transform=transforms, specific=True)
    pm = PredictionMapper(data_path, data_path + 'predictions/{}_{}'.format(radius, npoints), radius)

    for hc in ch.hc_names:
        merged = None
        for batch in tqdm(range(math.ceil(ch.get_hybrid_length(hc) / batch_size))):
            t_pts = torch.zeros((batch_size, npoints, 3))
            t_features = torch.ones((batch_size, npoints, 1))
            centroids = []

            for j in range(batch_size):
                chunk, centroid = ch[(hc, batch*batch_size+j)]
                centroids.append(centroid)
                t_pts[j] = torch.from_numpy(chunk.vertices)

            # apply model to batch of samples
            t_pts = t_pts.to(device, non_blocking=True)
            t_features = t_features.to(device, non_blocking=True)
            start = time.time()
            outputs = model(t_features, t_pts)
            model_time += time.time() - start
            model_counter += 1

            t_pts = t_pts.cpu().detach().numpy()
            output_np = outputs.cpu().detach().numpy()
            output_np = np.argmax(output_np, axis=2)

            for j in range(batch_size):
                if not np.all(t_pts[j] == 0):
                    prediction = PointCloud(t_pts[j], output_np[j])

                    # Apply invers transformations
                    prediction.move(centroids[j])
                    prediction.scale(radius)

                    if merged is None:
                        merged = prediction
                    else:
                        merged = clouds.merge_clouds(merged, prediction)

                    pm.map_predictions(prediction, hc, batch*batch_size+j)

        pm.save_prediction()
        clouds.save_cloud(merged, data_path + 'predictions/{}_{}'.format(radius, npoints), '{}_merged'.format(hc))

    model_time = model_time / model_counter
    print(model_time)


if __name__ == '__main__':
    radius = 15000
    npoints = 5000

    args = ['/u/jklimesch/thesis/trainings/current/',                   # save_root
            '/u/jklimesch/thesis/gt/gt_poisson/ads/single/',            # train_path
            radius,                                                     # radius
            npoints,                                                    # npoints
            '2020_01_14_{}'.format(radius) + '_{}'.format(npoints),       # name
            3,                                                          # nclasses
            [clouds.Normalization(radius),
             clouds.Center()],
            16,                                                         # batch_size
            True,                                                       # use_cuda
            1,                                                          # input_channels
            True,                                                       # use_big
            0                                                           # random_seed
            ]

    validation_thread(args)
