import os
import glob
import time
import numpy as np
import sklearn.metrics as sm
from tqdm import tqdm
import open3d as o3d
from morphx.processing import objects
from morphx.data import basics
from typing import List, Tuple, Dict, Union
from neuronx.classes.argscontainer import ArgsContainer
from neuronx.pipeline import infer, analyse
from morphx.classes.cloudensemble import CloudEnsemble


# -------------------------------------- HELPER METHODS ------------------------------------------- #

def handle_unpreds(gt: np.ndarray, hc: np.ndarray, drop: bool) -> Tuple[np.ndarray, np.ndarray]:
    """ Removes labels which equal -1. """
    if drop:
        mask = np.logical_and(gt != -1, hc != -1)
        return gt[mask], hc[mask]
    else:
        return gt, hc


def get_target_names(gtl: np.ndarray, hcl: np.ndarray, targets: list) -> list:
    """ Extracts the names of the labels which appear in gtl and hcl. """
    targets = np.array(targets)
    total = np.unique(np.concatenate((gtl, hcl), axis=0)).astype(int)
    return list(targets[total])


def write_confusion_matrix(cm: np.array, names: list) -> str:
    txt = f"{'':<15}"
    for name in names:
        txt += f"{name:<15}"
    txt += '\n'
    for ix, name in enumerate(names):
        txt += f"{name:<15}"
        for num in cm[ix]:
            txt += f"{num:<15}"
        txt += '\n'
    return txt


# -------------------------------------- EVALUATION METHODS ------------------------------------------- #

def eval_validation_set(set_path: str, total=True, mode: str = 'mvs', filters: bool = False,
                        re_evaluation: bool = False, drop_unpreds: bool = True, data_type: str = 'ce',
                        eval_name: str = 'evaluation', targets: list = None,
                        label_mappings: List[Tuple[int, int]] = None, label_remove: List[int] = None):
    """ Evaluates validations from multiple trainings.

    Args:
        set_path: path to validation folders
        total: flag for generating a total evaluation
        mode: 'd': direct mode (first prediction is taken), 'mv': majority vote mode (majority vote on predictions)
            'mvs' majority vote smoothing mode (majority vote on predicitons and smoothing afterwards)
        filters: flag for applying filters to skeleton predictions
        re_evaluation: flag for forcing a reevaluation
        drop_unpreds: flag for removing vertices without predictions
        data_type: type of dataset ('ce' for CloudEnsembles, 'hc' for HybridClouds)
        eval_name: name of evaluation
        targets: target names
    """
    set_path = os.path.expanduser(set_path)
    dirs = os.listdir(set_path)
    reports = {}
    for di in tqdm(dirs):
        di_in_path = set_path + di + '/'
        di_out_path = set_path + di + '/' + eval_name + '/'
        if os.path.exists(di_out_path) and not re_evaluation:
            print(di + " has already been processed. Skipping...")
            continue
        print("Processing " + di)
        if not os.path.exists(di_in_path + 'argscont.pkl'):
            print(f'{di} has no argscont.pkl file and gets skipped...')
            continue
        argscont = ArgsContainer().load_from_pkl(di_in_path + 'argscont.pkl')
        report = eval_validation(di_in_path, di_out_path, argscont, report_name=eval_name, total=total, mode=mode,
                                 filters=filters, drop_unpreds=drop_unpreds, data_type=data_type,
                                 targets=targets, label_mappings=label_mappings, label_remove=label_remove)
        report.update(argscont.attr_dict)
        reports[di] = report
    basics.save2pkl(reports, set_path, name=eval_name)


def eval_validation(input_path: str, output_path: str, argscont: ArgsContainer, report_name: str = 'Evaluation',
                    total: bool = False, mode: str = 'mvs', filters: bool = False, drop_unpreds: bool = True,
                    data_type: str = 'ce', targets: list = None, label_mappings: List[Tuple[int, int]] = None,
                    label_remove: List[int] = None):
    """ Apply different metrics to HybridClouds with predictions and compare these predictions with corresponding
        ground truth files with different filters or under different conditions.

    Args:
        input_path: Location of HybridClouds with predictions, saved as pickle files by a MorphX prediction mapper.
        output_path: Location where results of evaluation should be saved.
        argscont: Argument container which was generated during training of the model.
        report_name: Name of the current evaluation. Is used as the filename in which the evaluation report gets saved.
        total: Combine the predictions of all files to apply the metrics to the total prediction array.
        mode: 'd': direct mode (first prediction is taken), 'mv': majority vote mode (majority vote on predictions)
            'mvs' majority vote smoothing mode (majority vote on predicitons and smoothing afterwards)
        filters: After mapping the vertex labels to the skeleton, this flag can be used to apply filters to the skeleton
            and append the evaluation of these filtered skeletons.
        drop_unpreds: Flag for dropping all vertices or nodes which don't have predictions and whose labels are thus set
            to -1. If this flag is not set, the number of vertices or nodes without predictions might be much higher
            than the one of the predicted vertices or nodes, which results in bad evaluation results.
        data_type: 'ce' for CloudEnsembles, 'hc' for HybridClouds
        targets: encoding of labels
    """
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(input_path + 'sso_*.pkl')
    reports = {}
    reports_txt = "Confusion matrix: row = true label, column = predicted label \n" \
                  "Precision: What percentage of points from one label truly belong to that label \n" \
                  "Recall: What percentage of points from one true label have been predicted as that label \n\n"
    # Arrays for concatenating the labels of all files for later total evaluation
    total_labels = {'pred': np.array([]), 'pred_node': np.array([]), 'gt': np.array([]), 'gt_node': np.array([]),
                    'coverage': [0, 0]}
    # Build single file evaluation reports
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        if label_remove is None:
            label_remove = argscont.label_remove
        if label_mappings is None:
            label_mappings = argscont.label_mappings
        report, report_txt = eval_obj(file, total_labels, mode=mode, target_names=targets, filters=filters,
                                      drop_unpreds=drop_unpreds, data_type=data_type,
                                      label_mapping=label_mappings, label_remove=label_remove)
        reports[name] = report
        reports_txt += name + '\n\n' + report_txt + '\n\n\n'
    # Perform evaluation on total label arrays (labels from all files sticked together), prediction
    # mappings or filters are already included
    total_report = {}
    total_report_txt = 'Total\n\n'
    if total:
        coverage = total_labels['coverage']
        total_report['cov'] = coverage
        targets = get_target_names(total_labels['gt'], total_labels['pred'], targets)
        total_report[mode] = \
            sm.classification_report(total_labels['gt'], total_labels['pred'], output_dict=True, target_names=targets)
        total_report_txt += \
            mode + '\n\n' + \
            f'Coverage: {coverage[1] - coverage[0]} of {coverage[1]}, ' \
            f'{round((1 - coverage[0] / coverage[1]) * 100)} %\n\n' + \
            sm.classification_report(total_labels['gt'], total_labels['pred'], target_names=targets) + '\n\n'
        cm = sm.confusion_matrix(total_labels['gt'], total_labels['pred'])
        total_report_txt += write_confusion_matrix(cm, targets) + '\n\n'
        mode += '_skel'
        if filters:
            mode += '_f'
        targets = get_target_names(total_labels['gt_node'], total_labels['pred_node'], targets)
        total_report[mode] = \
            sm.classification_report(total_labels['gt_node'], total_labels['pred_node'],
                                     output_dict=True, target_names=targets)
        total_report_txt += \
            mode + '\n\n' + \
            sm.classification_report(total_labels['gt_node'], total_labels['pred_node'], target_names=targets) + '\n\n'
        cm_skel = sm.confusion_matrix(total_labels['gt_node'], total_labels['pred_node'])
        total_report_txt += write_confusion_matrix(cm_skel, targets) + '\n\n'
    reports['total'] = total_report
    reports_txt += total_report_txt
    basics.save2pkl(reports, output_path, name=report_name)
    with open(output_path + report_name + '.txt', 'w') as f:
        f.write(reports_txt)
    return reports


def eval_obj(file: str, total: dict = None, mode: str = 'mvs', target_names: list = None, filters: bool = False,
             drop_unpreds: bool = True, data_type: str = 'obj',
             label_mapping: List[Tuple[int, int]] = None, label_remove: List[int] = None) -> tuple:
    """ Apply different metrics to HybridClouds with predictions and compare these predictions with corresponding
        ground truth files with different filters or under different conditions. See eval_dataset for argument
        description.

    Returns:
        Evaluation report as string.
    """
    reports = {}
    reports_txt = ""
    file = os.path.expanduser(file)
    # load predictions and corresponding ground truth
    preds = basics.load_pkl(file)
    obj = objects.load_obj(data_type, preds[0])
    if label_remove is not None:
        obj.remove_nodes(label_remove)
    obj.set_predictions(preds[1])
    reports['pred_num'] = obj.pred_num
    if label_mapping is not None:
        obj.map_labels(label_mapping)
    # Perform majority vote on existing predictions and set these as new labels
    start = time.time()
    if mode == 'd':
        obj.generate_pred_labels(False)
    elif mode == 'mv':
        obj.generate_pred_labels()
    elif mode == 'mvs':
        obj.generate_pred_labels()
        obj.hc.prediction_smoothing()
    else:
        raise ValueError(f"Mode {mode} is not known.")
    pred2label_timing = time.time() - start
    reports[mode + '_timing'] = pred2label_timing
    if isinstance(obj, CloudEnsemble):
        hc = obj.hc
    else:
        hc = obj
    if len(hc.pred_labels) != len(hc.labels):
        raise ValueError("Length of predicted label array doesn't match with length of label array.")
    # Get evaluation for vertices
    coverage = hc.get_coverage()
    reports['cov'] = coverage
    # remove unpredicted labels
    gtl, hcl = handle_unpreds(hc.labels, hc.pred_labels, drop_unpreds)
    # generate evaluation and save it as pkl and as txt
    targets = get_target_names(gtl, hcl, target_names)
    reports[mode] = sm.classification_report(gtl, hcl, output_dict=True, target_names=targets)
    reports_txt += mode + '\n\n' \
                   + f'Coverage: {coverage[1] - coverage[0]} of {coverage[1]}, ' \
                     f'{round((1-coverage[0]/coverage[1])*100)} %\n\n' + \
                   f'Timing: {round(pred2label_timing, 3)} s\n\n' + \
                   f'Number of predictions: {obj.pred_num}\n\n' + \
                   sm.classification_report(gtl, hcl, target_names=targets) + '\n\n'
    cm = sm.confusion_matrix(gtl, hcl)
    reports_txt += write_confusion_matrix(cm, targets) + '\n\n'
    # Get evaluation for skeletons
    start = time.time()
    mode += '_skel'
    if filters:
        # apply smoothing filters
        hc.clean_node_labels(neighbor_num=10)
        mode += '_f'
    # remove unpredicted labels
    gtnl, hcnl = handle_unpreds(hc.node_labels, hc.pred_node_labels, drop_unpreds)
    map2skel_timing = time.time() - start
    reports[mode + '_timing'] = map2skel_timing
    # generate evaluation and save it as pkl and as txt
    targets = get_target_names(gtnl, hcnl, target_names)
    reports[mode] = sm.classification_report(gtnl, hcnl, output_dict=True, target_names=targets)
    reports_txt += mode + '\n\n' + f'Timing: {round(map2skel_timing, 3)} s\n\n' + \
                   sm.classification_report(gtnl, hcnl, target_names=targets) + '\n\n'
    cm = sm.confusion_matrix(gtnl, hcnl)
    reports_txt += write_confusion_matrix(cm, targets) + '\n\n'
    # save generated labels for total evaluation
    if total is not None:
        total['pred'] = np.append(total['pred'], hcl)
        total['pred_node'] = np.append(total['pred_node'], hcnl)
        total['gt'] = np.append(total['gt'], gtl)
        total['gt_node'] = np.append(total['gt_node'], gtnl)
        total['coverage'][0] += coverage[0]
        total['coverage'][1] += coverage[1]
    return reports, reports_txt


# -------------------------------------- PIPELINE METHODS ------------------------------------------- #

def full_evaluation_pipe(set_path: str, val_path, total=True, mode: str = 'mv', filters: bool = False,
                         drop_unpreds: bool = True, data_type: str = 'ce', eval_name: str = 'evaluation',
                         pipe_steps=None, val_iter=2, batch_num: int = -1, save_worst_examples: bool = False,
                         val_type: str = 'training_set', model_freq: Union[int, list] = 1, target_names: List[str] = None,
                         re_evaluation: bool = False, specific_model: int = None, redundancy: int = -1,
                         force_split: bool = False, model_max: int = None, label_mappings: List[Tuple[int, int]] = None,
                         label_remove: List[int] = None, same_seeds: bool = False):
    """ Runs full pipeline on given training set including validation and evaluation.

    Args:
        val_type: 'training_set' for using the 'validate_training_set' validation method or 'multiple_model' for
            using the 'validate_multi_model_training' validation method.
    """
    if pipe_steps is None:
        pipe_steps = [True, True]
    out_path = set_path + f'{eval_name}_valiter{val_iter}_batchsize{batch_num}/'
    eval_name += f'_{mode}'
    set_path = os.path.expanduser(set_path)
    val_path = os.path.expanduser(val_path)
    out_path = os.path.expanduser(out_path)
    if save_worst_examples:
        cloud_out_path = out_path
    else:
        cloud_out_path = None
    if pipe_steps[0]:
        # run validations
        if val_type == 'training_set':
            infer.validate_training_set(set_path, val_path, out_path, model_type='state_dict.pth', val_iter=val_iter,
                                        batch_num=batch_num, cloud_out_path=cloud_out_path, redundancy=redundancy,
                                        force_split=force_split)
        elif val_type == 'multiple_model':
            infer.validate_multi_model_training(set_path, val_path, out_path, model_freq, val_iter=val_iter,
                                                batch_num=batch_num, cloud_out_path=cloud_out_path,
                                                specific_model=specific_model, redundancy=redundancy,
                                                force_split=force_split, model_max=model_max,
                                                label_mappings=label_mappings, label_remove=label_remove,
                                                same_seeds=same_seeds)
        else:
            raise ValueError("val_type not known.")
    if pipe_steps[1]:
        # evaluate validations
        eval_validation_set(out_path, total=total, mode=mode, filters=filters, drop_unpreds=drop_unpreds,
                            data_type=data_type, eval_name=eval_name, targets=target_names, re_evaluation=re_evaluation,
                            label_mappings=label_mappings, label_remove=label_remove)


if __name__ == '__main__':
    # start full pipeline
    s_path = '~/thesis/current_work/paper/7_class/2020_08_26_5000_5000/'
    # v_path = '/u/jklimesch/thesis/gt/20_06_09/voxeled/evaluation/'
    v_path = '/u/jklimesch/thesis/gt/cmn/dnh/voxeled/evaluation/'
    # target_names = ['dendrite', 'neck', 'head']
    # target_names = ['dendrite', 'spine']
    # target_names = ['shaft', 'other', 'neck', 'head']
    target_names = ['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head']

    eval_name = 'eval'
    full_evaluation_pipe(s_path, v_path, eval_name=eval_name, pipe_steps=[True, True], val_iter=1, batch_num=-1,
                         save_worst_examples=False, val_type='multiple_model', model_freq=50, model_max=800,
                         target_names=target_names, redundancy=1)

    report_name = eval_name + '_mv'
    o_path = s_path + eval_name + '_valiter1_batchsize-1/'
    analyse.summarize_reports(o_path, report_name)
    r_path = o_path + report_name + '.pkl'
    analyse.generate_diagrams(r_path, o_path, [''], [''], points=False, density=False, part_key='mv',
                              filter_identifier=False, neg_identifier=[], time=True)
