import os
import glob
import math
import torch
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
from torch import nn
from typing import List, Tuple
from morphx.processing import clouds, basics
from neuronx.classes.torchhandler import TorchHandler
from morphx.classes.pointcloud import PointCloud
from morphx.postprocessing.mapping import PredictionMapper
from elektronn3.models.convpoint import SegAdapt, SegBig
from neuronx.classes.argscontainer import ArgsContainer
from lightconvpoint.utils import get_network


@torch.no_grad()
def validate_single(th: TorchHandler, hc: str, batch_size: int, point_num: int, iter_num: int,
                    device: torch.device, model, pm: PredictionMapper, input_channels: int,
                    out_path: str = None, sampling: bool = True, lcp_flag: bool = False):
    """ Can be used to validate single objects. Returns timing for chunk generation, prediction and mapping. """
    chunk_timing = [0, 0]
    model_timing = [0, 0]
    map_timing = [0, 0]
    batch_num = math.ceil(th.get_obj_length(hc) / batch_size)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(iter_num):
        for batch in tqdm(range(batch_num)):
            if not sampling:
                start = time.time()
                sample = th[(hc, batch * batch_size)]
                chunk_timing[0] += time.time() - start
                chunk_timing[1] += 1
                point_num = len(sample['pts'])
            pts = torch.zeros((batch_size, point_num, 3))
            features = torch.ones((batch_size, point_num, input_channels))
            mapping_idcs = torch.ones((batch_size, point_num))
            o_mask = torch.zeros((batch_size, point_num, th.num_classes))
            l_mask = torch.zeros((batch_size, point_num))
            targets = torch.zeros((batch_size, point_num))
            fill_up = 0
            sample_number = 0
            remove = []
            for j in range(batch_size):
                # for sampling == False, the batch_size is always 1 as the samples have different sizes.
                if sampling:
                    start = time.time()
                    sample = th[(hc, batch * batch_size + j)]
                    chunk_timing[0] += time.time() - start
                    chunk_timing[1] += 1
                    # fill up empty batches (happening when all parts of current cell have been processed).
                    # The fill up samples are always build by the first parts of the current cell (thus fill_up = 0)
                    # and will be removed later
                    if torch.all(sample['pts'] == 0):
                        if sample_number == 0:
                            sample_number = j
                        sample = th[(hc, fill_up)]
                        fill_up = (fill_up + 1) % sample_number
                        remove.append(j)
                pts[j] = sample['pts']
                features[j] = sample['features']
                mapping_idcs[j] = sample['map']
                o_mask[j] = sample['o_mask']
                l_mask[j] = sample['l_mask']
                targets[j] = sample['target']

            # apply model to batch of samples
            if lcp_flag:
                pts = pts.transpose(1, 2)
                features = features.transpose(1, 2)

            pts = pts.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            start = time.time()
            outputs = model(features, pts)
            model_timing[0] += time.time() - start
            model_timing[1] += 1

            if lcp_flag:
                pts = pts.transpose(1, 2)
                features = features.transpose(1, 2)
                outputs = outputs.transpose(1, 2)

            # convert all tensors to numpy arrays
            pts = pts.cpu().detach().numpy()
            features = features.cpu().detach().numpy()

            l_mask = l_mask.numpy().astype(bool)
            o_mask = o_mask.numpy().astype(bool)
            targets = targets.numpy()
            mapping_idcs = mapping_idcs.numpy()
            output_np = outputs.cpu().detach().numpy()

            # save bad examples
            if out_path is not None:
                t_loss = 0
                worst_ix = 0
                for j in range(batch_size):
                    if np.random.random() < 1 and '80035' in hc:
                        if j not in remove:
                            curr_output = output_np[j][o_mask[j]].reshape(-1, th.num_classes)
                            curr_target = targets[j][l_mask[j]].astype(int)
                            loss = criterion(torch.from_numpy(curr_output), torch.from_numpy(curr_target))
                            if loss > t_loss:
                                worst_ix = j
                                t_loss = loss
                            curr_output = np.argmax(curr_output, axis=1)
                            target_curr = PointCloud(pts[j], targets[j], features=features[j])
                            output_curr = PointCloud(pts[j][l_mask[j].astype(bool)], curr_output)
                            curr = [target_curr, output_curr]
                            basics.save2pkl(curr, out_path, f'{hc}_i{i}_idx{batch * batch_size + j}')
                # worst_output = np.argmax(output_np[worst_ix][o_mask[worst_ix]].reshape(-1, th.num_classes), axis=1)
                # target_cloud = PointCloud(pts[worst_ix], targets[worst_ix])
                # output_cloud = PointCloud(pts[worst_ix][l_mask[worst_ix].astype(bool)], worst_output)
                # worst = [target_cloud, output_cloud]
                # basics.save2pkl(worst, out_path, f'{hc}_i{i}_b{batch}_i{worst_ix}')

            # apply argmax to outputs
            output_np = np.argmax(output_np, axis=2)

            for j in range(batch_size):
                if j not in remove:
                    start = time.time()
                    # filter the points of the outputs which should get a prediction
                    curr_pts = pts[j]
                    curr_out = output_np[j]
                    curr_map = mapping_idcs[j]
                    curr_mask = l_mask[j]
                    curr_pts = curr_pts[curr_mask]
                    curr_out = curr_out[curr_mask]
                    curr_map = curr_map[curr_mask]
                    # map predictions to original cloud
                    prediction = PointCloud(curr_pts, curr_out)
                    pm.map_predictions(prediction, curr_map, hc, batch * batch_size + j, sampling=sampling)
                    map_timing[0] += time.time() - start
                    map_timing[1] += 1

    return chunk_timing[0] / chunk_timing[1], model_timing[0] / model_timing[1], map_timing[0] / map_timing[1]


def validation(argscont: ArgsContainer, training_path: str, val_path: str, out_path: str,
               model_type: str = 'state_dict.pth', val_iter: int = 1, batch_num: int = -1,
               cloud_out_path: str = None, redundancy: int = -1, force_split: bool = False, same_seeds: bool = False,
               label_mappings: List[Tuple[int, int]] = None, label_remove: List[int] = None,
               border_exclusion: int = 0):
    training_path = os.path.expanduser(training_path)
    val_path = os.path.expanduser(val_path)
    out_path = os.path.expanduser(out_path)
    if os.path.exists(out_path):
        print(f"{out_path} already exists. Skipping...")
        return

    if same_seeds:
        # set random seeds to ensure compareability
        torch.manual_seed(argscont.random_seed)
        np.random.seed(argscont.random_seed)
        random.seed(argscont.random_seed)

    if argscont.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    lcp_flag = False
    # load model
    if argscont.architecture == 'lcp' or argscont.model == 'ConvAdaptSeg':
        kwargs = {}
        if argscont.model == 'ConvAdaptSeg':
            kwargs = dict(f_map_num=argscont.pl, architecture=argscont.architecture, act=argscont.act,
                          norm=argscont.norm_type)
        conv = dict(layer=argscont.conv[0], kernel_separation=argscont.conv[1])
        model = get_network(argscont.model, argscont.input_channels, argscont.class_num, conv, argscont.search, **kwargs)
        lcp_flag = True
    elif argscont.use_big:
        model = SegBig(argscont.input_channels, argscont.class_num, trs=argscont.track_running_stats, dropout=0,
                       use_bias=argscont.use_bias, norm_type=argscont.norm_type, use_norm=argscont.use_norm,
                       kernel_size=argscont.kernel_size, neighbor_nums=argscont.neighbor_nums,
                       reductions=argscont.reductions, first_layer=argscont.first_layer,
                       padding=argscont.padding, nn_center=argscont.nn_center, centroids=argscont.centroids,
                       pl=argscont.pl, normalize=argscont.cp_norm)
    else:
        print("Adaptable model was found!")
        model = SegAdapt(argscont.input_channels, argscont.class_num, architecture=argscont.architecture,
                         trs=argscont.track_running_stats, dropout=argscont.dropout, use_bias=argscont.use_bias,
                         norm_type=argscont.norm_type, kernel_size=argscont.kernel_size, padding=argscont.padding,
                         nn_center=argscont.nn_center, centroids=argscont.centroids, kernel_num=argscont.pl,
                         normalize=argscont.cp_norm, act=argscont.act)
    try:
        full = torch.load(training_path + model_type)
        model.load_state_dict(full)
    except RuntimeError:
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

    if redundancy == -1:
        redundancy = argscont.splitting_redundancy
    if label_remove is None:
        label_remove = argscont.label_remove
    if label_mappings is None:
        label_mappings = argscont.label_mappings

    th = TorchHandler(val_path, argscont.sample_num, argscont.class_num, density_mode=argscont.density_mode,
                      bio_density=argscont.bio_density, tech_density=argscont.tech_density, transform=transforms,
                      specific=True, obj_feats=argscont.features, ctx_size=argscont.chunk_size,
                      label_mappings=label_mappings, hybrid_mode=argscont.hybrid_mode,
                      feat_dim=argscont.input_channels, splitting_redundancy=redundancy,
                      label_remove=label_remove, sampling=argscont.sampling,
                      force_split=force_split, padding=argscont.padding, exclude_borders=border_exclusion)
    pm = PredictionMapper(val_path, out_path, th.splitfile, label_remove=label_remove, hybrid_mode=argscont.hybrid_mode)

    if batch_num == -1:
        batch_size = argscont.batch_size
    else:
        batch_size = batch_num

    attr_dicts = {obj: None for obj in th.obj_names}
    # perform validation
    obj = None
    obj_names = th.obj_names.copy()
    for obj in th.obj_names:
        # skip trainings where validation has already been generated
        if os.path.exists(out_path + obj + '_preds.pkl'):
            print(obj + " has already been processed. Skipping...")
            obj_names.remove(obj)
            continue
        if th.get_obj_length(obj) == 0:
            print(obj + " has no chunks to process. Skipping...")
            obj_names.remove(obj)
            continue
        print(f"Processing {obj}")
        attr_dict = th.get_obj_info(obj, hybrid_only=True)
        start = time.time()
        chunk_timing, model_timing, map_timing = \
            validate_single(th, obj, batch_size, argscont.sample_num, val_iter, device, model, pm,
                            argscont.input_channels, out_path=cloud_out_path, sampling=argscont.sampling,
                            lcp_flag=lcp_flag)
        total_timing = time.time() - start
        attr_dict['timing'] = total_timing
        attr_dicts[obj] = attr_dict
        total_times.append(total_timing)
        chunk_times.append(chunk_timing)
        model_times.append(model_timing)
        map_times.append(map_timing)
    if obj is not None:
        pm.save_prediction()
    else:
        return 

    # save timing results and object information
    if os.path.exists(out_path + 'obj_info.pkl'):
        with open(out_path + 'obj_info.pkl', 'rb') as f:
            attr_dict = pickle.load(f)
            attr_dict.update(attr_dicts)
            pickle.dump(attr_dict, f)
        f.close()
    else:
        with open(out_path + 'obj_info.pkl', 'wb') as f:
            pickle.dump(attr_dicts, f)
        f.close()

    with open(out_path + 'timing.txt', 'a') as f:
        f.write('\nModel timing:\n\n')
        for idx, item in enumerate(obj_names):
            f.write(f'{item}: \t\t {model_times[idx]} s.\n')
        f.write('\nChunk timing:\n\n')
        for idx, item in enumerate(obj_names):
            f.write(f'{item}: \t\t {chunk_times[idx]} s.\n')
        f.write('\nMapping timing:\n\n')
        for idx, item in enumerate(obj_names):
            f.write(f'{item}: \t\t {map_times[idx]} s.\n')
        f.write('\nTotal timing:\n\n')
        for idx, item in enumerate(obj_names):
            f.write(f'{item}: \t\t {total_times[idx]} s.\n')
        f.close()

    argscont.save2pkl(out_path + 'argscont.pkl')

    # free CUDA memory
    del model
    torch.cuda.empty_cache()


def validate_training_set(set_path: str, val_path: str, out_path: str, model_type: str = 'state_dict.pth',
                          val_iter: int = 1, batch_num: int = -1, cloud_out_path: str = None, redundancy: int = -1,
                          force_split: bool = False):
    """ Validate multiple trainings.

    Args:
        set_path: path where the trainings are located.
        val_path: path to cell files on which the trained models should get validated.
        out_path: path where validation folders should get saved.
        model_type: name of model file which should be used.
        val_iter: number of validation iterations.
        batch_num: Batch size in inference mode can be larger than during training. Default is same as during training.
        cloud_out_path: Path to save worst inference examples
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
            print("No arguments found for this training. Skipping...")
            continue
        if cloud_out_path is not None:
            curr_out_path = cloud_out_path + di + '/examples/'
        else:
            curr_out_path = None
        validation(argscont, set_path + di + '/', val_path, out_path + di + '/', model_type=model_type,
                   val_iter=val_iter, batch_num=batch_num, cloud_out_path=curr_out_path, redundancy=redundancy,
                   force_split=force_split)


def validate_multi_model_training(training_path: str, val_path: str, out_path: str, model_freq: int,
                                  model_min: int = None, val_iter: int = 1, batch_num: int = -1,
                                  cloud_out_path: str = None, specific_model: int = None, redundancy: int = -1,
                                  force_split: bool = False, model_max: int = None,
                                  label_mappings: List[Tuple[int, int]] = None, label_remove: List[int] = None,
                                  same_seeds: bool = False, border_exclusion: int = 0):
    """ Can be used to validate every model_freq file where all the models are saved in set_path as torch state dicts
        with the format: 'state_dict_e{epoch_number}.pth'.

    Args:
        training_path: path where the trainings are located.
        val_path: path to cell files on which the trained models should get validated.
        out_path: path where validation folder should get saved.
        model_freq: Every model_freq state_dict at the set_path gets evaluated.
        val_iter: number of validation iterations.
        batch_num: Batch size in inference mode can be larger than during training. Default is same as during training.
        cloud_out_path: Path to save worst inference examples.
    """
    training_path = os.path.expanduser(training_path)
    val_path = os.path.expanduser(val_path)
    out_path = os.path.expanduser(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # load argument container
    if os.path.exists(training_path + '/argscont.pkl'):
        argscont = ArgsContainer().load_from_pkl(training_path + '/argscont.pkl')
    else:
        print("No argument container found for this training.")
        return
    # prepare for saving worst examples
    if cloud_out_path is not None:
        curr_out_path = cloud_out_path + '/examples/'
    else:
        curr_out_path = None
    # validate different models
    model_path = training_path + 'models/'
    if not os.path.exists(model_path):
        print("Model folder was not found in training. The folder must be named 'models'.")
        return
    models = glob.glob(model_path + 'state_dict_*')
    models.sort()
    if specific_model is None:
        if model_min is None:
            model_min = 0
        if model_max is None:
            model_max = 500
        model_idcs = np.arange(model_min, model_max, model_freq)
        for ix in model_idcs:
            model_type = f'state_dict_e{ix}.pth'
            if curr_out_path is not None:
                curr_out_path += f'epoch_{ix}/'
            validation(argscont, model_path, val_path, out_path + f'epoch_{ix}' + '/', model_type=model_type,
                       val_iter=val_iter, batch_num=batch_num, cloud_out_path=curr_out_path, redundancy=redundancy,
                       force_split=force_split, label_mappings=label_mappings, label_remove=label_remove,
                       same_seeds=same_seeds, border_exclusion=border_exclusion)
    else:
        model_type = f'state_dict_e{specific_model}.pth'
        validation(argscont, model_path, val_path, out_path + f'epoch_{specific_model}' + '/', model_type=model_type,
                   val_iter=val_iter, batch_num=batch_num, cloud_out_path=curr_out_path, redundancy=redundancy,
                   force_split=force_split, label_mappings=label_mappings, label_remove=label_remove,
                   same_seeds=same_seeds, border_exclusion=border_exclusion)

