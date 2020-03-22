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
from neuronx.classes.argscontainer import ArgsContainer, args2container_14


@torch.no_grad()
def validate_single(th: TorchHandler, hc: str, batch_size: int, point_num: int, iter_num: int,
                    device: torch.device, model, pm: PredictionMapper, input_channels: int):
    """ Can be used to validate single objects. Returns timing for chunk generation, prediction and mapping. """
    chunk_timing = [0, 0]
    model_timing = [0, 0]
    map_timing = [0, 0]
    batch_num = math.ceil(th.get_obj_length(hc) / batch_size)
    for i in range(iter_num):
        for batch in tqdm(range(batch_num)):
            pts = torch.zeros((batch_size, point_num, 3))
            features = torch.ones((batch_size, point_num, input_channels))
            mapping_idcs = torch.ones((batch_size, point_num))
            mask = torch.zeros((batch_size, point_num))
            fill_up = 0
            remove = []
            for j in range(batch_size):
                start = time.time()
                sample = th[(hc, batch * batch_size + j)]
                chunk_timing[0] += time.time() - start
                chunk_timing[1] += 1
                # fill up empty batches (happening when all parts of current cell have been processed).
                # The fill up samples are always build by the first parts of the current cell (thus fill_up = 0)
                # and will be removed later
                if torch.all(sample['pts'] == 0):
                    sample = th[(hc, fill_up)]
                    fill_up += 1
                    remove.append(j)
                pts[j] = sample['pts']
                features[j] = sample['features']
                mapping_idcs[j] = sample['map']
                mask[j] = sample['l_mask']

            # apply model to batch of samples
            pts = pts.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            start = time.time()
            outputs = model(features, pts)
            model_timing[0] += time.time() - start
            model_timing[1] += 1

            # convert all tensors to numpy arrays and apply argmax to outputs
            pts = pts.cpu().detach().numpy()
            mask = mask.numpy()
            mapping_idcs = mapping_idcs.numpy()
            output_np = outputs.cpu().detach().numpy()
            output_np = np.argmax(output_np, axis=2)

            for j in range(batch_size):
                if j not in remove:
                    start = time.time()
                    # filter the points of the outputs which should get a prediction
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
                    map_timing[0] += time.time() - start
                    map_timing[1] += 1
    return chunk_timing[0] / chunk_timing[1], model_timing[0] / model_timing[1], map_timing[0] / map_timing[1]


def validation(argscont: ArgsContainer, training_path: str, val_path: str, out_path: str,
               model_type: str = 'state_dict.pth', val_iter: int = 1, batch_num: int = -1):
    training_path = os.path.expanduser(training_path)
    val_path = os.path.expanduser(val_path)
    out_path = os.path.expanduser(out_path)

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
    full = torch.load(training_path + model_type)
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
                      specific=True, obj_feats=argscont.features, chunk_size=argscont.chunk_size,
                      label_mappings=argscont.label_mappings, hybrid_mode=argscont.hybrid_mode)
    pm = PredictionMapper(val_path, out_path, th.splitfile)

    if batch_num == -1:
        batch_size = argscont.batch_size
    else:
        batch_size = batch_num

    # perform validation
    obj = None
    for obj in th.obj_names:
        # skip trainings where validation has already been generated
        if os.path.exists(out_path + obj + '_preds.pkl'):
            print(obj + " has already been processed. Skipping...")
            obj = None
            continue
        print(f"Processing {obj}")
        start = time.time()
        chunk_timing, model_timing, map_timing = \
            validate_single(th, obj, batch_size, argscont.sample_num, val_iter, device, model, pm,
                            argscont.input_channels)
        total_timing = time.time() - start
        total_times.append(total_timing)
        chunk_times.append(chunk_timing)
        model_times.append(model_timing)
        map_times.append(map_timing)
    if obj is not None:
        pm.save_prediction()
    else:
        return 

    # save timing results
    with open(out_path + 'timing.txt', 'a') as f:
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

    argscont.save2pkl(out_path + 'argscont.pkl')

    # free CUDA memory
    del model
    torch.cuda.empty_cache()


def validate_training_set(set_path: str, val_path: str, out_path: str, model_type: str = 'state_dict.pth',
                          val_iter: int = 1, batch_num: int = -1):
    """ Validate multiple trainings.

    Args:
        set_path: path where the trainings are located.
        val_path: path to cell files on which the trained models should get validated.
        out_path: path where validation folders should get saved.
        model_type: name of model file which should be used.
        val_iter: number of validation iterations.
        batch_num: Batch size in inference mode can be larger than during training. Default is same as during training.
    """
    set_path = os.path.expanduser(set_path)
    val_path = os.path.expanduser(val_path)
    out_path = os.path.expanduser(out_path)
    dirs = os.listdir(set_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for di in dirs:
        print(f"Processing {di}")
        if not os.path.isdir(set_path + di):
            continue
        if os.path.exists(set_path + di + '/argscont.pkl'):
            argscont = ArgsContainer().load_from_pkl(set_path + di + '/argscont.pkl')
        else:
            if os.path.exists(set_path + di + '/training_args.pkl'):
                argscont = args2container_14(set_path + di + '/training_args.pkl')
            else:
                print("No arguments found for this training. Skipping...")
                continue
        validation(argscont, set_path + di + '/', val_path, out_path + di + '/', model_type=model_type,
                   val_iter=val_iter, batch_num=batch_num)
