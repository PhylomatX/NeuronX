import os
import open3d
import glob
import numpy as np
import sklearn.metrics as sm
from tqdm import tqdm
from morphx.processing import objects, basics
from typing import List, Tuple, Union
from neuronx.classes.argscontainer import ArgsContainer
from neuronx.pipeline import predict
from morphx.classes.cloudensemble import CloudEnsemble


# -------------------------------------- HELPER METHODS ------------------------------------------- #

def get_target_names(labels: np.ndarray, predictions: np.ndarray, label_names: list) -> list:
    """ Extracts the names of the labels which appear in gtl and hcl. """
    label_names = np.array(label_names)
    total = np.unique(np.concatenate((labels, predictions), axis=0)).astype(int)
    return list(label_names[total])


def write_confusion_matrix(matrix: np.array, label_names: list) -> str:
    txt = f"{'':<15}"
    for name in label_names:
        txt += f"{name:<15}"
    txt += '\n'
    for ix, name in enumerate(label_names):
        txt += f"{name:<15}"
        for num in matrix[ix]:
            txt += f"{num:<15}"
        txt += '\n'
    return txt


# -------------------------------------- EVALUATION METHODS ------------------------------------------- #

def evaluate_prediction_set(set_path: str,
                            total_evaluation=True,
                            evaluation_mode: str = 'mv',
                            skeleton_smoothing: bool = False,
                            force_evaluation: bool = False,
                            remove_unpredicted: bool = True,
                            data_type: str = 'ce',
                            report_name: str = 'evaluation',
                            label_names: list = None,
                            label_mappings: List[Tuple[int, int]] = None,
                            label_remove: List[int] = None):
    """ 
    Can be used to evaluate multiple prediction folders.

    Args:
        set_path: path to prediction folders.
        total_evaluation: generate total evaluation (merging all single cell evaluations).
        evaluation_mode: 'd': direct mode (first prediction is taken),
              'mv': majority vote mode (majority vote on predictions)
              'mvs' majority vote smoothing mode (majority vote on predicitons and smoothing afterwards)
        skeleton_smoothing: Smooth skeleton predictions with sliding window.
        force_evaluation: Overwrite existing evaluations.
        remove_unpredicted: remove points without predictions.
        data_type: type of dataset ('ce' for CloudEnsembles, 'hc' for HybridClouds).
        report_name: name of evaluation report.
        label_names: target names of classes.
        label_mappings: List of tuples like (from, to) where 'from' is label which should get mapped to 'to'.
            Defaults to label_mappings from training or to val_label_mappings of ArgsContainer.
        label_remove: List of labels to remove from the cells.
            Defaults to label_remove from training or to val_label_remove of ArgsContainer.
    """
    set_path = os.path.expanduser(set_path)
    folders = os.listdir(set_path)
    reports = {}
    for folder in tqdm(folders):
        prediction_folder = set_path + folder + '/'
        evaluation_folder = set_path + folder + '/' + report_name + '/'
        if os.path.exists(evaluation_folder) and not force_evaluation:
            print(folder + " has already been processed. Skipping...")
            continue
        print("Processing " + folder)
        if not os.path.exists(prediction_folder + 'argscont.pkl'):
            print(f'{folder} has no argscont.pkl file and gets skipped...')
            continue
        argscont = ArgsContainer().load_from_pkl(prediction_folder + 'argscont.pkl')
        report = evaluate_model_predictions(prediction_folder, evaluation_folder, argscont, report_name=report_name,
                                            total_evaluation=total_evaluation, evaluation_mode=evaluation_mode, skeleton_smoothing=skeleton_smoothing,
                                            remove_unpredicted=remove_unpredicted, data_type=data_type,
                                            label_names=label_names, label_mappings=label_mappings, label_remove=label_remove)
        report.update(argscont.attr_dict)
        reports[folder] = report
    basics.save2pkl(reports, set_path, name=report_name)


def evaluate_model_predictions(prediction_folder: str,
                               evaluation_folder: str,
                               argscont: ArgsContainer = None,
                               report_name: str = 'evaluation',
                               total_evaluation: bool = False,
                               evaluation_mode: str = 'mv',
                               skeleton_smoothing: bool = False,
                               remove_unpredicted: bool = True,
                               data_type: str = 'ce',
                               label_names: list = None,
                               label_mappings: List[Tuple[int, int]] = None,
                               label_remove: List[int] = None):
    """
    Can be used to evaluate a single prediction folder. See `evaluate_prediction_set` for argument descriptions.
    """
    prediction_folder = os.path.expanduser(prediction_folder)
    evaluation_folder = os.path.expanduser(evaluation_folder)
    predictions = glob.glob(prediction_folder + '*_preds.pkl')
    reports = {}
    reports_txt = "Confusion matrix: row = true label, column = predicted label \n" \
                  "Precision: What percentage of points from one label truly belong to that label \n" \
                  "Recall: What percentage of points from one true label have been predicted as that label \n\n"

    # arrays for concatenating the labels of all files for later total evaluation
    total_labels = {'vertex_predictions': np.array([]),
                    'node_predictions': np.array([]),
                    'vertex_labels': np.array([]),
                    'node_labels': np.array([]),
                    'coverage': [0, 0]}

    # --- build single file evaluation reports ---
    for prediction in tqdm(predictions):
        slashs = [pos for pos, char in enumerate(prediction) if char == '/']
        name = prediction[slashs[-1] + 1:-4]
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
        report, report_txt = evaluate_cell_predictions(prediction, total_labels, evaluation_mode=evaluation_mode, label_names=label_names,
                                                       skeleton_smoothing=skeleton_smoothing, remove_unpredicted=remove_unpredicted, data_type=data_type,
                                                       label_mapping=label_mappings, label_remove=label_remove)
        reports[name] = report
        reports_txt += name + '\n\n' + report_txt + '\n\n\n'

    # --- build total evaluation report (by merging all single file labels and evaluating the result).
    # Filters and label mappings are already included in the single file labels. ---
    total_report = {}
    total_report_txt = 'Total\n\n'
    if total_evaluation:
        coverage = total_labels['coverage']
        total_report['cov'] = coverage

        # --- vertex evaluation ---
        targets_vertex = get_target_names(total_labels['vertex_labels'], total_labels['vertex_predictions'], label_names)
        total_report[evaluation_mode] = \
            sm.classification_report(total_labels['vertex_labels'], total_labels['vertex_predictions'], output_dict=True,
                                     target_names=targets_vertex)
        total_report_txt += \
            evaluation_mode + '\n\n' + \
            f'Coverage: {coverage[1] - coverage[0]} of {coverage[1]}, ' \
            f'{round((1 - coverage[0] / coverage[1]) * 100)} %\n\n' + \
            sm.classification_report(total_labels['vertex_labels'], total_labels['vertex_predictions'], target_names=targets_vertex) + '\n\n'
        confusion_matrix = sm.confusion_matrix(total_labels['vertex_labels'], total_labels['vertex_predictions'])
        total_report_txt += write_confusion_matrix(confusion_matrix, targets_vertex) + '\n\n'

        # --- skeleton evaluation ---
        evaluation_mode += '_skel'
        targets_node = get_target_names(total_labels['node_labels'], total_labels['node_predictions'], label_names)
        total_report[evaluation_mode] = \
            sm.classification_report(total_labels['node_labels'], total_labels['node_predictions'],
                                     output_dict=True, target_names=targets_node)
        total_report_txt += \
            evaluation_mode + '\n\n' + \
            sm.classification_report(total_labels['node_labels'], total_labels['node_predictions'],
                                     target_names=targets_node) + '\n\n'
        skeleton_confusion_matrix = sm.confusion_matrix(total_labels['node_labels'], total_labels['node_predictions'])
        total_report_txt += write_confusion_matrix(skeleton_confusion_matrix, targets_node) + '\n\n'

    # --- write reports to file ---
    reports['total'] = total_report
    reports_txt += total_report_txt
    basics.save2pkl(reports, evaluation_folder, name=report_name)
    with open(evaluation_folder + report_name + '.txt', 'w') as f:
        f.write(reports_txt)
    return reports


def evaluate_cell_predictions(prediction: str,
                              total_evaluation: dict = None,
                              evaluation_mode: str = 'mv',
                              label_names: list = None,
                              skeleton_smoothing: bool = False,
                              remove_unpredicted: bool = True,
                              data_type: str = 'obj',
                              label_mapping: List[Tuple[int, int]] = None,
                              label_remove: List[int] = None) -> tuple:
    """
    Can be used to evaluation single cell prediction. See `evaluate_prediction_set` for argument descriptions.
    """
    reports = {}
    reports_txt = ""
    prediction = os.path.expanduser(prediction)
    # --- merge predictions and corresponding ground truth (predictions contain pointer to original cell file)
    vertex_predictions = basics.load_pkl(prediction)
    obj = objects.load_obj(data_type, vertex_predictions[0])
    if label_remove is not None:
        obj.remove_nodes(label_remove)
    obj.set_predictions(vertex_predictions[1])
    reports['pred_num'] = obj.pred_num
    if label_mapping is not None:
        obj.map_labels(label_mapping)

    # --- reduce multiple predictions to single label by either:
    # d: taking first prediction as label
    # mv: taking majority vote on all predictions as new label
    # mvs: taking majority vote on all predictions as label and apply smoothing ---
    if evaluation_mode == 'd':
        obj.generate_pred_labels(False)
    elif evaluation_mode == 'mv':
        obj.generate_pred_labels()
    elif evaluation_mode == 'mvs':
        obj.generate_pred_labels()
        obj.hc.prediction_smoothing()
    else:
        raise ValueError(f"Mode {evaluation_mode} is not known.")

    if isinstance(obj, CloudEnsemble):
        hc = obj.hc
    else:
        hc = obj
    if len(hc.pred_labels) != len(hc.labels):
        raise ValueError("Length of predicted label array doesn't match with length of label array.")

    coverage = hc.get_coverage()
    reports['cov'] = coverage

    # --- vertex evaluation ---
    vertex_labels = hc.labels
    vertex_predictions = hc.pred_labels
    if remove_unpredicted:
        mask = np.logical_and(vertex_labels != -1, vertex_predictions != -1)
        vertex_labels, vertex_predictions = vertex_labels[mask], vertex_predictions[mask]
    targets = get_target_names(vertex_labels, vertex_predictions, label_names)
    reports[evaluation_mode] = sm.classification_report(vertex_labels, vertex_predictions, output_dict=True, target_names=targets)
    reports_txt += evaluation_mode + '\n\n' + \
                     f'Coverage: {coverage[1] - coverage[0]} of {coverage[1]}, ' \
                     f'{round((1 - coverage[0] / coverage[1]) * 100)} %\n\n' + \
                     f'Number of predictions: {obj.pred_num}\n\n' + \
                     sm.classification_report(vertex_labels, vertex_predictions, target_names=targets) + '\n\n'
    cm = sm.confusion_matrix(vertex_labels, vertex_predictions)
    reports_txt += write_confusion_matrix(cm, targets) + '\n\n'

    # --- skeleton evaluation ---
    evaluation_mode += '_skel'
    if skeleton_smoothing:
        hc.node_sliding_window_bfs(neighbor_num=20)
    node_labels = hc.node_labels
    node_predictions = hc.pred_node_labels
    if remove_unpredicted:
        mask = np.logical_and(node_labels != -1, node_predictions != -1)
        node_labels, node_predictions = node_labels[mask], node_predictions[mask]
    targets = get_target_names(node_labels, node_predictions, label_names)
    reports[evaluation_mode] = sm.classification_report(node_labels, node_predictions, output_dict=True, target_names=targets)
    reports_txt += evaluation_mode + '\n\n' + sm.classification_report(node_labels, node_predictions, target_names=targets) + '\n\n'
    cm = sm.confusion_matrix(node_labels, node_predictions)
    reports_txt += write_confusion_matrix(cm, targets) + '\n\n'

    # --- save generated labels for total evaluation ---
    if total_evaluation is not None:
        total_evaluation['vertex_predictions'] = np.append(total_evaluation['vertex_predictions'], vertex_predictions)
        total_evaluation['node_predictions'] = np.append(total_evaluation['node_predictions'], node_predictions)
        total_evaluation['vertex_labels'] = np.append(total_evaluation['vertex_labels'], vertex_labels)
        total_evaluation['node_labels'] = np.append(total_evaluation['node_labels'], node_labels)
        total_evaluation['coverage'][0] += coverage[0]
        total_evaluation['coverage'][1] += coverage[1]
    return reports, reports_txt


def predict_and_evaluate(train_path: str,
                         cell_path,
                         total_evaluation=True,
                         evaluation_mode: str = 'mv',
                         skeleton_smoothing: bool = False,
                         remove_unpredicted: bool = True,
                         data_type: str = 'ce',
                         report_name: str = 'evaluation',
                         prediction_redundancy: int = 2,
                         batch_size: int = -1,
                         model_freq: Union[int, list] = 1,
                         model_max: int = None,
                         model_min: int = None,
                         label_names: List[str] = None,
                         force_evaluation: bool = False,
                         specific_model: int = None,
                         chunk_redundancy: int = -1,
                         force_split: bool = False,
                         label_mappings: List[Tuple[int, int]] = None,
                         label_remove: List[int] = None,
                         training_seed: bool = False,
                         border_exclusion: int = 0,
                         model=None):
    """
    Runs full pipeline on given training set including validation and evaluation.
    See `generate_predictions_from_training` and `evaluate_prediction_set` for argument descriptions.
    """
    out_path = train_path + f'{report_name}/'
    report_name += f'_{evaluation_mode}'
    train_path = os.path.expanduser(train_path)
    cell_path = os.path.expanduser(cell_path)
    out_path = os.path.expanduser(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(out_path + 'eval_kwargs.txt', 'w') as f:
        f.write(str(locals()))

    predict.generate_predictions_from_training(train_path, cell_path, out_path, model_freq, model_min=model_min,
                                               prediction_redundancy=prediction_redundancy, batch_size=batch_size,
                                               specific_model=specific_model, chunk_redundancy=chunk_redundancy,
                                               force_split=force_split, model_max=model_max, label_mappings=label_mappings,
                                               label_remove=label_remove, training_seed=training_seed,
                                               border_exclusion=border_exclusion, model=model)

    evaluate_prediction_set(out_path, total_evaluation=total_evaluation, evaluation_mode=evaluation_mode, skeleton_smoothing=skeleton_smoothing,
                            remove_unpredicted=remove_unpredicted, data_type=data_type, report_name=report_name, label_names=label_names,
                            force_evaluation=force_evaluation, label_mappings=label_mappings, label_remove=label_remove)
