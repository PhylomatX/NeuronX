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


def validate_single(ch: ChunkHandler, hc: str, batch_size: int, point_num: int, iter_num: int,
                    device: torch.device, model, pm: PredictionMapper):
    """ Can be used to validate single objects. Returns timing for chunk generation, prediction and mapping. """
    chunk_timing = 0
    model_timing = 0
    map_timing = 0

    batch_num = math.ceil(ch.get_obj_length(hc) / batch_size)

    zero_counter = 0
    for i in range(iter_num):
        for batch in tqdm(range(batch_num)):
            pts = torch.zeros((batch_size, point_num, 3))
            feats = torch.ones((batch_size, point_num, 1))
            mapping_idcs = np.ones((batch_size, point_num))
            obj_bounds = []

            for j in range(batch_size):
                start = time.time()
                chunk, idcs = ch[(hc, batch * batch_size + j)]
                obj_bounds.append(chunk.obj_bounds)
                chunk_timing += time.time() - start
                mapping_idcs[j] = idcs
                pts[j] = torch.from_numpy(chunk.vertices)

            # apply model to batch of samples
            pts = pts.to(device, non_blocking=True)
            feats = feats.to(device, non_blocking=True)
            start = time.time()
            outputs = model(feats, pts)
            model_timing += time.time() - start

            pts = pts.cpu().detach().numpy()
            output_np = outputs.cpu().detach().numpy()
            output_np = np.argmax(output_np, axis=2)

            for j in range(batch_size):
                if not np.all(pts[j] == 0):
                    start = time.time()
                    prediction = PointCloud(pts[j], output_np[j], obj_bounds=obj_bounds[j])
                    pm.map_predictions(prediction, mapping_idcs[j], hc, batch * batch_size + j)
                    map_timing += time.time() - start
                else:
                    zero_counter += 1

    chunk_factor = batch_size * iter_num * batch_num
    map_factor = batch_size * iter_num * batch_num - zero_counter
    return chunk_timing / chunk_factor, model_timing / (iter_num * batch_num), map_timing / map_factor


def validation_thread(args):
    """ Can be used for parallel validations using the multiprocessing framework from SyConn. """

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
    model.eval()

    # load scripted model
    # model_path = save_root + '/' + name + '/model.pts'
    # model = torch.jit.load(model_path, map_location=device)
    # model.eval()

    chunk_times = []
    map_times = []
    model_times = []

    transforms = clouds.Compose(transforms)
    ch = ChunkHandler(data_path, radius, npoints, transform=transforms, specific=True)
    pm = PredictionMapper(data_path, save_root + name + '/predictions/', radius)

    info_folder = save_root + name + '/predictions/info/'
    if not os.path.exists(info_folder):
        os.makedirs(info_folder)

    with open(info_folder + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)
        f.close()

    for hc in ch.obj_names:
        chunk_timing, model_timing, map_timing = validate_single(ch, hc, batch_size, npoints, iter_num, device, model, pm)
        chunk_times.append(chunk_timing)
        model_times.append(model_timing)
        map_times.append(map_timing)
    pm.save_prediction()

    with open(info_folder + 'timing.txt', 'w') as f:
        f.write('\nModel timing:\n\n')
        for idx, item in enumerate(ch.obj_names):
            f.write(f'{item}: \t\t {model_times[idx]} s.\n')
        f.write('\nChunk timing:\n\n')
        for idx, item in enumerate(ch.obj_names):
            f.write(f'{item}: \t\t {chunk_times[idx]} s.\n')
        f.write('\nMapping timing:\n\n')
        for idx, item in enumerate(ch.obj_names):
            f.write(f'{item}: \t\t {map_times[idx]} s.\n')
        f.close()


if __name__ == '__main__':
    radius = 15000
    npoints = 10000

    for radius, npoints in [(20000, 8192), (15000, 6000), (10000, 4096)]:
        args = ['/u/jklimesch/thesis/trainings/past/2019/12_17/',           # save_root
                '/u/jklimesch/thesis/gt/gt_poisson/',                       # train_path
                radius,                                                     # radius
                npoints,                                                    # npoints
                '{}'.format(radius) + '_{}'.format(npoints),                # name
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
