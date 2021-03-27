import os
import math
import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from morphx.processing import clouds
from neuronx.classes.torchhandler import TorchHandler
from morphx.classes.pointcloud import PointCloud
from morphx.postprocessing.mapping import PredictionMapper
from neuronx.classes.argscontainer import ArgsContainer
from lightconvpoint.utils.network import get_search, get_conv
from elektronn3.models.lcp_adapt import ConvAdaptSeg


@torch.no_grad()
def predict_cell(data_loader: TorchHandler,
                 cell: str,
                 batch_size: int,
                 point_num: int,
                 prediction_redundancy: int,
                 device: torch.device,
                 model,
                 prediction_mapper: PredictionMapper,
                 input_channels: int,
                 point_subsampling: bool = True):
    """
    Can be used to generate predictions for single cells using a pre-loaded model.

    Args:
        data_loader: see TorchHandler class.
        cell: cell identifier.
        batch_size: batch size.
        point_num: number of points in each chunk.
        prediction_redundancy: number of times each cell should be processed (using the same chunks but different points due to random sampling).
        device: cuda or cpu.
        model: pre-loaded PyTorch model.
        prediction_mapper: See PredictionMapper class.
        input_channels: number of input features.
        point_subsampling: sample random points from extracted cell chunks.
    """
    batch_num = math.ceil(data_loader.get_obj_length(cell) / batch_size)

    for i in range(prediction_redundancy):
        for batch in tqdm(range(batch_num)):
            # --- prepare batches ---
            if not point_subsampling:
                # for point_subsampling == False, the batch_size is always 1 as the samples have different sizes
                # => determine the number of points in the next chunk.
                cell_chunk = data_loader[(cell, batch * batch_size)]
                point_num = len(cell_chunk['pts'])
            pts = torch.zeros((batch_size, point_num, 3))
            features = torch.ones((batch_size, point_num, input_channels))
            mapping_idcs = torch.ones((batch_size, point_num))
            prediction_mask = torch.zeros((batch_size, point_num))
            targets = torch.zeros((batch_size, point_num))

            # --- fill batches ---
            fill_up_ix = 0
            fill_up_start_ix = 0
            remove = []
            for j in range(batch_size):
                if point_subsampling:
                    cell_chunk = data_loader[(cell, batch * batch_size + j)]
                    # fill up empty batches with first chunks from current cell
                    if torch.all(cell_chunk['pts'] == 0):
                        if fill_up_start_ix == 0:
                            fill_up_start_ix = j
                        cell_chunk = data_loader[(cell, fill_up_ix)]
                        fill_up_ix = (fill_up_ix + 1) % fill_up_start_ix
                        remove.append(j)
                pts[j] = cell_chunk['pts']
                features[j] = cell_chunk['features']
                mapping_idcs[j] = cell_chunk['map']
                prediction_mask[j] = cell_chunk['l_mask']
                targets[j] = cell_chunk['target']

            # --- apply model to batches (LightConvPoint requires transpose) ---
            pts = pts.transpose(1, 2)
            features = features.transpose(1, 2)
            pts = pts.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            outputs = model(features, pts)
            pts = pts.transpose(1, 2)
            outputs = outputs.transpose(1, 2)

            pts = pts.cpu().detach().numpy()
            prediction_mask = prediction_mask.numpy().astype(bool)
            mapping_idcs = mapping_idcs.numpy()
            output_np = outputs.cpu().detach().numpy()
            output_np = np.argmax(output_np, axis=2)

            # --- map chunk predictions back to full cell ---
            for j in range(batch_size):
                # remove chunks which were used to fill up the batch
                if j not in remove:
                    # filter the points of the outputs which should get a prediction
                    curr_mask = prediction_mask[j]
                    curr_pts = pts[j][curr_mask]
                    curr_out = output_np[j][curr_mask]
                    curr_map = mapping_idcs[j][curr_mask]
                    prediction = PointCloud(curr_pts, curr_out)
                    prediction_mapper.map_predictions(prediction, curr_map, cell, batch * batch_size + j,
                                                      sampling=point_subsampling)


def generate_predictions_with_model(argscont: ArgsContainer,
                                    model_path: str,
                                    cell_path: str,
                                    out_path: str,
                                    prediction_redundancy: int = 1,
                                    batch_size: int = -1,
                                    chunk_redundancy: int = -1,
                                    force_split: bool = False,
                                    training_seed: bool = False,
                                    label_mappings: List[Tuple[int, int]] = None,
                                    label_remove: List[int] = None,
                                    border_exclusion: int = 0,
                                    state_dict: str = None,
                                    model=None,
                                    **args):
    """
    Can be used to generate predictions for multiple files using a specific model (either passed as path to state_dict or as pre-loaded model).

    Args:
        argscont: argument container for current model.
        model_path: path to model state dict.
        cell_path: path to cells used for prediction.
        out_path: path to folder where predictions of this model should get saved.
        prediction_redundancy: number of times each cell should be processed (using the same chunks but different points due to random sampling).
        batch_size: batch size, if -1 this defaults to the batch size used during training.
        chunk_redundancy: number of times each cell should get splitted into a complete chunk set (including different chunks each time).
        force_split: split cells even if cached split information exists.
        training_seed: use random seed from training.
        label_mappings: List of tuples like (from, to) where 'from' is label which should get mapped to 'to'.
            Defaults to label_mappings from training or to val_label_mappings of ArgsContainer.
        label_remove: List of labels to remove from the cells. Defaults to label_remove from training or to val_label_remove of ArgsContainer.
        border_exclusion: nm distance which defines how much of the chunk borders should be excluded from predictions.
        state_dict: state dict holding model for prediction.
        model: loaded model to use for prediction.
    """
    if os.path.exists(out_path):
        print(f"{out_path} already exists. Skipping...")
        return

    if training_seed:
        torch.manual_seed(argscont.random_seed)
        np.random.seed(argscont.random_seed)
        random.seed(argscont.random_seed)

    if argscont.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if model is not None:
        model = model
    else:
        kwargs = {}
        if argscont.model == 'ConvAdaptSeg':
            kwargs = dict(kernel_num=argscont.pl, architecture=argscont.architecture, activation=argscont.act,
                          norm=argscont.norm_type)
        conv = dict(layer=argscont.conv[0], kernel_separation=argscont.conv[1])
        model = ConvAdaptSeg(argscont.input_channels, argscont.class_num, get_conv(conv), get_search(argscont.search),
                             **kwargs)
        try:
            full = torch.load(model_path + state_dict)
            model.load_state_dict(full)
        except RuntimeError:
            model.load_state_dict(full['model_state_dict'])
        model.to(device)
        model.eval()

    transforms = clouds.Compose(argscont.val_transforms)
    if chunk_redundancy == -1:
        chunk_redundancy = argscont.splitting_redundancy
    if batch_size == -1:
        batch_size = argscont.batch_size
    if label_remove is None:
        if argscont.val_label_remove is not None:
            label_remove = argscont.val_label_remove
        else:
            label_remove = argscont.label_remove
    if label_mappings is None:
        if argscont.val_label_mappings is not None:
            label_mappings = argscont.val_label_mappings
        else:
            label_mappings = argscont.label_mappings

    torch_handler = TorchHandler(cell_path, argscont.sample_num, argscont.class_num, density_mode=argscont.density_mode,
                                 bio_density=argscont.bio_density, tech_density=argscont.tech_density,
                                 transform=transforms, specific=True, obj_feats=argscont.features, ctx_size=argscont.chunk_size,
                                 label_mappings=label_mappings, hybrid_mode=argscont.hybrid_mode,
                                 feat_dim=argscont.input_channels, splitting_redundancy=chunk_redundancy,
                                 label_remove=label_remove, sampling=argscont.sampling,
                                 force_split=force_split, padding=argscont.padding, exclude_borders=border_exclusion)
    prediction_mapper = PredictionMapper(cell_path, out_path, torch_handler.splitfile, label_remove=label_remove,
                                         hybrid_mode=argscont.hybrid_mode)

    obj = None
    obj_names = torch_handler.obj_names.copy()
    for obj in torch_handler.obj_names:
        if os.path.exists(out_path + obj + '_preds.pkl'):
            print(obj + " has already been processed. Skipping...")
            obj_names.remove(obj)
            continue
        if torch_handler.get_obj_length(obj) == 0:
            print(obj + " has no chunks to process. Skipping...")
            obj_names.remove(obj)
            continue
        print(f"Processing {obj}")
        predict_cell(torch_handler, obj, batch_size, argscont.sample_num, prediction_redundancy, device, model,
                     prediction_mapper, argscont.input_channels, point_subsampling=argscont.sampling)
    if obj is not None:
        prediction_mapper.save_prediction()
    else:
        return
    argscont.save2pkl(out_path + 'argscont.pkl')
    del model
    torch.cuda.empty_cache()


def generate_predictions_from_training(train_path: str,
                                       cell_path: str,
                                       out_path: str,
                                       model_freq: int,
                                       model_min: int = 0,
                                       model_max: int = 500,
                                       specific_model: int = None,
                                       model=None,
                                       **args):
    """
    Can be used to generate predictions for the files in `cell_path` using models of the training specified by `train_path`.

    Three modes:
        + multi-model prediction: Generate predictions for all models within epoch range (specific_model, model = None)
        + single-model prediction: Generate predictions using specific epoch model (model = None, specific_model = epoch number)
        + loaded model prediction: Generate predictions using model that has already been loaded (model = loaded PyTorch model)

    For multi-model and single-model prediction, `train_path` must contain a 'models' folder where models exist as state dicts
    in the format: 'state_dict_e{epoch_number}.pth'

    Args:
        train_path: path to training folder.
        cell_path: path to cells used for prediction.
        out_path: path where predictions should be saved.
        model_freq: predictions are generated with each model in epoch range (model_min, model_max, model_freq).
        model_min: lower bound of model_freq.
        model_max: higher bound of model_freq.
        specific_model: epoch number of model which should be used for prediction.
        model: loaded model to use for prediction.
    """
    train_path = os.path.expanduser(train_path)
    cell_path = os.path.expanduser(cell_path)
    out_path = os.path.expanduser(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if os.path.exists(train_path + '/argscont.pkl'):
        argscont = ArgsContainer().load_from_pkl(train_path + '/argscont.pkl')
    else:
        warnings.warn("No argument container found. Skipping.")
        return
    if model is not None:
        # inference with model that has already been loaded and passed as 'model'
        generate_predictions_with_model(argscont, '', cell_path, out_path + f'epoch_{specific_model}' + '/',
                                        model=model, **args)
    else:
        # inference using models from 'models' folder
        model_path = train_path + 'models/'
        if not os.path.exists(model_path):
            warnings.warn("Model folder does not exist. The folder must be named 'models'. Skipping.")
            return
        if specific_model is not None:
            # inference with specific model from 'models' folder, represented by its epoch number
            state_dict = f'state_dict_e{specific_model}.pth'
            generate_predictions_with_model(argscont, model_path, cell_path, out_path + f'epoch_{specific_model}' + '/',
                                            state_dict=state_dict, **args)
        else:
            # inference with all models from 'models' folder which are in the range described by model_min, model_max, model_freq
            model_idcs = np.arange(model_min, model_max, model_freq)
            for ix in model_idcs:
                state_dict = f'state_dict_e{ix}.pth'
                generate_predictions_with_model(argscont, model_path, cell_path, out_path + f'epoch_{ix}' + '/',
                                                state_dict=state_dict, **args)
