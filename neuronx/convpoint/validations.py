import os
import math
import torch
import time
import random
import pickle
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
    iter_num = args[12]

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

    full = torch.load(save_root + name + '/state_dict.pth')
    model.load_state_dict(full['model_state_dict'])
    model.to(device)
    # model.eval()

    # load scripted model
    # model_path = save_root + '/' + name + '/model.pts'
    # model = torch.jit.load(model_path, map_location=device)
    # model.eval()

    hc_timing = []
    mapping_timing = []
    model_time = 0
    model_counter = 0

    transforms = clouds.Compose(transforms)
    ch = ChunkHandler(data_path, radius, npoints, transform=transforms, specific=True)
    pm = PredictionMapper(data_path, save_root + name + '/predictions/', radius)

    info_folder = save_root + name + '/predictions/info/'
    if not os.path.exists(info_folder):
        os.makedirs(info_folder)

    with open(info_folder + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)
        f.close()

    for hc in ch.hc_names:
        curr_hc_timing = 0
        curr_map_timing = 0
        for i in range(iter_num):
            for batch in tqdm(range(math.ceil(ch.get_hybrid_length(hc) / batch_size))):
                hc_start = time.time()
                pts = torch.zeros((batch_size, npoints, 3))
                feats = torch.ones((batch_size, npoints, 1))
                mapping_idcs = np.ones((batch_size, npoints))

                for j in range(batch_size):
                    chunk, idcs = ch[(hc, batch*batch_size+j)]
                    mapping_idcs[j] = idcs
                    pts[j] = torch.from_numpy(chunk.vertices)

                # apply model to batch of samples
                pts = pts.to(device, non_blocking=True)
                feats = feats.to(device, non_blocking=True)
                start = time.time()
                outputs = model(feats, pts)
                model_time += time.time() - start
                model_counter += 1

                pts = pts.cpu().detach().numpy()
                output_np = outputs.cpu().detach().numpy()
                output_np = np.argmax(output_np, axis=2)

                curr_hc_timing += time.time() - hc_start

                map_start = time.time()
                for j in range(batch_size):
                    if not np.all(pts[j] == 0):
                        prediction = PointCloud(pts[j], output_np[j])
                        pm.map_predictions(prediction, mapping_idcs[j], hc, batch*batch_size+j)
                curr_map_timing += time.time() - map_start
        hc_timing.append(curr_hc_timing)
        mapping_timing.append(curr_map_timing)
    pm.save_prediction()

    model_time = model_time / model_counter
    with open(info_folder + 'timing.txt', 'w') as f:
        f.write(f'Convpoint timing, {batch_size * npoints} in {batch_size} batches: {model_time} s.\n')
        f.write('\nPrediction timing:\n\n')
        for idx, item in enumerate(ch.hc_names):
            f.write(f'{item}: \t\t {hc_timing[idx]} s.\n')
        f.write('\nMapping timing:\n\n')
        for idx, item in enumerate(ch.hc_names):
            f.write(f'{item}: \t\t {mapping_timing[idx]} s.\n')
        f.close()


if __name__ == '__main__':
    radius = 15000
    npoints = 10000

    for radius, npoints in [(20000, 8192), (15000, 6000), (10000, 4096)]:
        args = ['/u/jklimesch/thesis/trainings/past/2019/12_17/',           # save_root
                '/u/jklimesch/thesis/gt/gt_poisson/',                   # train_path
                radius,                                                     # radius
                npoints,                                                    # npoints
                '{}'.format(radius) + '_{}'.format(npoints),     # name
                5,                                                          # nclasses
                [clouds.Normalization(radius), clouds.Center()],
                16,                                                         # batch_size
                True,                                                       # use_cuda
                1,                                                          # input_channels
                True,                                                       # use_big
                0,                                                          # random_seed
                2                                                           # iteration number
                ]

        validation_thread(args)
