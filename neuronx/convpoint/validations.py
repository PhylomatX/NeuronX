import os
import math
import torch
import time
import random
import pickle
import numpy as np
from tqdm import tqdm
from morphx.processing import clouds
from morphx.data.torchhandler import TorchHandler
from morphx.classes.pointcloud import PointCloud
from morphx.postprocessing.mapping import PredictionMapper
from elektronn3.models.convpoint import SegSmall, SegBig


def validate_single(th: TorchHandler, hc: str, batch_size: int, point_num: int, iter_num: int,
                    device: torch.device, model, pm: PredictionMapper, input_channels: int):
    """ Can be used to validate single objects. Returns timing for chunk generation, prediction and mapping. """
    chunk_timing = 0
    model_timing = 0
    map_timing = 0

    batch_num = math.ceil(th.get_obj_length(hc) / batch_size)

    zero_counter = 0
    for i in range(iter_num):
        for batch in tqdm(range(batch_num)):
            pts = torch.zeros((batch_size, point_num, 3))
            features = torch.ones((batch_size, point_num, input_channels))
            mapping_idcs = torch.ones((batch_size, point_num))
            mask = torch.zeros((batch_size, point_num))

            for j in range(batch_size):
                start = time.time()
                sample = th[(hc, batch * batch_size + j)]
                chunk_timing += time.time() - start

                pts[j] = sample['pts']
                features[j] = sample['features']
                mapping_idcs[j] = sample['map']
                mask[j] = sample['l_mask']

            # apply model to batch of samples
            pts = pts.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            start = time.time()
            outputs = model(features, pts)
            model_timing += time.time() - start

            # convert all tensors to numpy arrays and apply argmax to outputs
            pts = pts.cpu().detach().numpy()
            mask = mask.numpy()
            mapping_idcs = mapping_idcs.numpy()
            output_np = outputs.cpu().detach().numpy()
            output_np = np.argmax(output_np, axis=2)

            for j in range(batch_size):
                if not np.all(pts[j] == 0):
                    start = time.time()

                    # filter the points their outputs which should get a prediction
                    curr_pts = pts[j]
                    curr_out = output_np[j]
                    curr_map = mapping_idcs[j]
                    curr_mask = mask[j].astype(bool)
                    curr_pts = curr_pts[curr_mask]
                    curr_out = curr_out[curr_mask]
                    curr_map = curr_map[curr_mask]

                    # map predictions to original cloud
                    prediction = PointCloud(curr_pts, curr_out)
                    pm.map_predictions(prediction, curr_map, hc, batch * batch_size + j)
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
    obj_feats = args[13]
    tech_density = args[14]
    bio_density = args[15]
    density_mode = args[16]
    folder = args[17]

    # set random seeds to ensure compareability of different trainings
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load model
    if use_big:
        model = SegBig(input_channels, nclasses)
    else:
        model = SegSmall(input_channels, nclasses)
    full = torch.load(save_root + name + '/state_dict.pth')
    model.load_state_dict(full['model_state_dict'])
    model.to(device)
    # load scripted model
    # model_path = save_root + '/' + name + '/model.pts'
    # model = torch.jit.load(model_path, map_location=device)
    model.eval()

    # set up environment
    chunk_times = []
    map_times = []
    model_times = []
    transforms = clouds.Compose(transforms)
    th = TorchHandler(data_path, npoints, nclasses, density_mode=density_mode, bio_density=bio_density,
                      tech_density=tech_density, transform=transforms, specific=True, obj_feats=obj_feats)
    pm = PredictionMapper(data_path, f'{save_root}{name}/{folder}/{radius}', th.splitfile)
    info_folder = f'{save_root}{name}/{folder}/info/'
    if not os.path.exists(info_folder):
        os.makedirs(info_folder)
    with open(info_folder + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)
        f.close()

    # perform validation
    for hc in th.obj_names:
        chunk_timing, model_timing, map_timing = \
            validate_single(th, hc, batch_size, npoints, iter_num, device, model, pm, input_channels)
        chunk_times.append(chunk_timing)
        model_times.append(model_timing)
        map_times.append(map_timing)
    pm.save_prediction()

    # save timing results
    with open(info_folder + 'timing.txt', 'w') as f:
        f.write('\nModel timing:\n\n')
        for idx, item in enumerate(th.obj_names):
            f.write(f'{item}: \t\t {model_times[idx]} s.\n')
        f.write('\nChunk timing:\n\n')
        for idx, item in enumerate(th.obj_names):
            f.write(f'{item}: \t\t {chunk_times[idx]} s.\n')
        f.write('\nMapping timing:\n\n')
        for idx, item in enumerate(th.obj_names):
            f.write(f'{item}: \t\t {map_times[idx]} s.\n')
        f.close()


if __name__ == '__main__':
    chunk_size = 15000
    sample_num = 15000
    args = ['/u/jklimesch/thesis/trainings/current/',                   # save_root
            '/u/jklimesch/thesis/gt/gt_meshsets/voxeled/',              # data path
            chunk_size,                                                 # radius
            sample_num,                                                 # npoints
            '2020_02_25_' + '{}'.format(chunk_size) +
            '_{}'.format(sample_num),                                   # name
            7,                                                          # nclasses
            [clouds.Normalization(chunk_size), clouds.Center()],
            4,                                                          # batch_size
            True,                                                       # use_cuda
            4,                                                          # input_channels
            True,                                                       # use_big
            0,                                                          # random_seed
            1,                                                          # iteration number
            {'hc': np.array([1, 0, 0, 0]),
             'mi': np.array([0, 1, 0, 0]),
             'vc': np.array([0, 0, 1, 0]),
             'sy': np.array([0, 0, 0, 1])
             },                                                         # features
            1500,                                                       # tech_density
            100,                                                        # bio_density
            True,                                                       # density_mode
            'predictions_tr'                                            # folder
            ]
    validation_thread(args)
