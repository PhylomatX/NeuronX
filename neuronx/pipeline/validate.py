import os
import math
import torch
import time
import random
import numpy as np
from tqdm import tqdm
from morphx.processing import clouds
from morphx.data.torchhandler import TorchHandler
from morphx.classes.pointcloud import PointCloud
from morphx.postprocessing.mapping import PredictionMapper
from elektronn3.models.convpoint import SegSmall, SegBig
from neuronx.classes.argscontainer import ArgsContainer, pkl2container


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


def validation(argscont: ArgsContainer, val_path: str):
    val_path = os.path.expanduser(val_path)

    # set random seeds to ensure compareability of different trainings
    torch.manual_seed(argscont.random_seed)
    np.random.seed(argscont.random_seed)
    random.seed(argscont.random_seed)

    if argscont.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load model
    if argscont.use_big:
        model = SegBig(argscont.input_channels, argscont.class_num)
    else:
        model = SegSmall(argscont.input_channels, argscont.class_num)
    full = torch.load(argscont.train_save_path + 'state_dict.pth')
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
    total_times = []
    transforms = clouds.Compose(argscont.val_transforms)
    th = TorchHandler(val_path, argscont.sample_num, argscont.class_num, density_mode=argscont.density_mode,
                      bio_density=argscont.bio_density, tech_density=argscont.tech_density, transform=transforms,
                      specific=True, obj_feats=argscont.features, chunk_size=argscont.chunk_size)
    pm = PredictionMapper(val_path, argscont.val_save_path, th.splitfile)

    # perform validation
    hc = None
    for hc in th.obj_names:
        start = time.time()
        chunk_timing, model_timing, map_timing = \
            validate_single(th, hc, argscont.batch_size, argscont.sample_num, argscont.val_iter, device, model, pm,
                            argscont.input_channels)
        total_timing = time.time() - start
        total_times.append(total_timing)
        chunk_times.append(chunk_timing)
        model_times.append(model_timing)
        map_times.append(map_timing)
    if hc is not None:
        pm.save_prediction(hc)

    # save timing results
    with open(argscont.val_info_path + 'timing.txt', 'w') as f:
        f.write('\nModel timing:\n\n')
        for idx, item in enumerate(th.obj_names):
            f.write(f'{item}: \t\t {model_times[idx]} s.\n')
        f.write('\nChunk timing:\n\n')
        for idx, item in enumerate(th.obj_names):
            f.write(f'{item}: \t\t {chunk_times[idx]} s.\n')
        f.write('\nMapping timing:\n\n')
        for idx, item in enumerate(th.obj_names):
            f.write(f'{item}: \t\t {map_times[idx]} s.\n')
        f.write('\nTotal timing:\n\n')
        for idx, item in enumerate(th.obj_names):
            f.write(f'{item}: \t\t {total_times[idx]} s.\n')
        f.close()

    argscont.info2pkl(argscont.val_info_path)

    # free CUDA memory
    del model
    torch.cuda.empty_cache()


def validate_training_set(set_path: str, val_path: str):
    """ Validate multiple trainings.

    Args:
        set_path: path where the trainings are located.
        val_path: path to cell files on which the trained models should get validated.
    """
    set_path = os.path.expanduser(set_path)
    dirs = os.listdir(set_path)
    for di in dirs:
        if os.path.exists(set_path + 'validation/' + di):
            print(di + " has already been processed. Skipping...")
            continue
        try:
            argscont = pkl2container(set_path + di + '/training_args.pkl')
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            continue
        validation(argscont, val_path)


if __name__ == '__main__':
    validate_training_set('/u/jklimesch/thesis/trainings/past/param_search_2/')
