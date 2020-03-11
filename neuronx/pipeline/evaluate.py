import os
import glob
from datetime import date
import numpy as np
import sklearn.metrics as sm
from tqdm import tqdm
from typing import Tuple
from morphx.processing import objects
from morphx.data import basics


def eval_dataset(input_path: str, gt_path: str, output_path: str, report_name: str = 'Evaluation',
                 total: bool = False, direct: bool = False, filters: bool = False, drop_unpreds: bool = True,
                 data_type: str = 'ce', target_names: list = None):
    """ Apply different metrics to HybridClouds with predictions and compare these predictions with corresponding
        ground truth files with different filters or under different conditions.

    Args:
        input_path: Location of HybridClouds with predictions, saved as pickle files by a MorphX prediction mapper.
        gt_path: Location of ground truth files, one for each file at input_path. Must have the same names as their
            counterparts.
        output_path: Location where results of evaluation should be saved.
        report_name: Name of the current evaluation. Is used as the filename in which the evaluation report gets saved.
        total: Combine the predictions of all files to apply the metrics to the total prediction array.
        direct: Flag for swichting off the majority vote which is normally applied if there are multiple predictions
            for one vertex. If this flag is set, only the first prediction is taken into account.
        filters: After mapping the vertex labels to the skeleton, this flag can be used to apply filters to the skeleton
            and append the evaluation of these filtered skeletons.
        drop_unpreds: Flag for dropping all vertices or nodes which don't have predictions and whose labels are thus set
            to -1. If this flag is not set, the number of vertices or nodes without predictions might be much higher
            than the one of the predicted vertices or nodes, which results in bad evaluation results.
        data_type: 'ce' for CloudEnsembles, 'hc' for HybridClouds
        target_names: encoding of labels
    """
    input_path = os.path.expanduser(input_path)
    gt_path = os.path.expanduser(gt_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(input_path + '*.pkl')
    gt_files = glob.glob(gt_path + '*.pkl')
    reports = {}
    reports_txt = ""
    # Arrays for concatenating the labels of all files for later total evaluation
    total_labels = {'pred': np.array([]), 'pred_node': np.array([]), 'gt': np.array([]), 'gt_node': np.array([])}
    # Build single file evaluation reports
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        # Find corresponding ground truth
        gt_file = None
        for item in gt_files:
            if name in item:
                gt_file = item
        if gt_file is None:
            print("Ground truth for {} was not found. Skipping file.".format(name))
            continue
        report, report_txt = eval_single(file, gt_file, total_labels, direct=direct, filters=filters,
                                         drop_unpreds=drop_unpreds, data_type=data_type, target_names=target_names)
        reports[name] = report
        reports_txt += name + '\n\n' + report_txt + '\n\n\n'
    # Perform evaluation on total label arrays (labels from all files sticked together), prediction
    # mappings or filters are already included
    total_report = {}
    total_report_txt = 'Total\n\n'
    if total:
        if direct:
            mode = 'd'
        else:
            mode = 'mv'
        targets = get_target_names(total_labels['gt'], total_labels['pred'], target_names)
        total_report[mode] = sm.classification_report(total_labels['gt'], total_labels['pred'], output_dict=True,
                                                      target_names=targets)
        total_report_txt += mode + '\n\n' + sm.classification_report(total_labels['gt'], total_labels['pred'],
                                                                     target_names=targets) + '\n\n'
        mode += '_skel'
        if filters:
            mode += '_f'
        targets = get_target_names(total_labels['gt_node'], total_labels['pred_node'], target_names)
        total_report[mode] = sm.classification_report(total_labels['gt_node'], total_labels['pred_node'],
                                                      output_dict=True, target_names=targets)
        total_report_txt += mode + '\n\n' + sm.classification_report(total_labels['gt_node'], total_labels['pred_node'],
                                                                     target_names=targets) + '\n\n'
    reports['total'] = total_report
    reports_txt += total_report_txt
    basics.save2pkl(reports, output_path, name=report_name)
    with open(output_path + report_name + '.txt', 'w') as f:
        f.write(reports_txt)
    return reports


def eval_single(file: str, gt_file: str, total: dict = None, direct: bool = False, target_names: list = None,
                filters: bool = False, drop_unpreds: bool = True, data_type: str = 'obj') -> tuple:
    """ Apply different metrics to HybridClouds with predictions and compare these predictions with corresponding
        ground truth files with different filters or under different conditions.

    Args:
        file: HybridCloud with predictions, saved as pickle file by a MorphX prediction mapper.
        gt_file: Ground truth file corresponding to the HybridCloud given in file.
        total: Use given dict to save processed predictions for later use (see eval_dataset).
        direct: Flag for swichting off the majority vote which is normally applied if there are multiple predictions
            for one vertex. If this flag is set, only the first prediction is taken into account.
        target_names: encoding of labels.
        filters: After mapping the vertex labels to the skeleton, this flag can be used to apply filters to the skeleton
            and append the evaluation of these filtered skeletons.
        drop_unpreds: Flag for dropping all vertices or nodes which don't have predictions and whose labels are thus set
            to -1. If this flag is not set, the number of vertices or nodes without predictions might be much higher
            than the one of the predicted vertices or nodes, which results in bad evaluation results.
        data_type: 'obj' for CloudEnsembles, 'hc' for HybridClouds

    Returns:
        Evaluation report as string.
    """
    file = os.path.expanduser(file)
    gt_file = os.path.expanduser(gt_file)
    # load HybridCloud and corresponding ground truth
    obj = objects.load_obj(data_type, file)
    gt_obj = objects.load_obj(data_type, gt_file)
    hc = obj.hc
    gt_hc = gt_obj.hc

    if len(hc.labels) != len(gt_hc.labels):
        raise ValueError("Length of ground truth label array doesn't match with length of predicted label array.")
    reports = {}
    reports_txt = ""
    # Perform majority vote on existing predictions and set these as new labels
    if direct:
        obj.preds2labels(False)
        mode = 'd'
    else:
        obj.preds2labels()
        mode = 'mv'
    # Get evaluation for vertices
    gtl, hcl = handle_unpreds(gt_hc.labels, hc.labels, drop_unpreds)

    targets = get_target_names(gtl, hcl, target_names)
    reports[mode] = sm.classification_report(gtl, hcl, output_dict=True,
                                             target_names=targets)
    reports_txt += mode + '\n\n' + sm.classification_report(gtl, hcl, target_names=targets) + '\n\n'
    # Get evaluation for skeletons
    mode += '_skel'
    if filters:
        hc.clean_node_labels()
        mode += '_f'
    gtnl, hcnl = handle_unpreds(gt_hc.node_labels, hc.node_labels, drop_unpreds)
    targets = get_target_names(gtnl, hcnl, target_names)
    reports[mode] = sm.classification_report(gtnl, hcnl, output_dict=True, target_names=targets)
    reports_txt += mode + '\n\n' + sm.classification_report(gtnl, hcnl, target_names=targets) + '\n\n'
    # save generated labels for total evaluation
    if total is not None:
        total['pred'] = np.append(total['pred'], hcl)
        total['pred_node'] = np.append(total['pred_node'], hcnl)
        total['gt'] = np.append(total['gt'], gtl)
        total['gt_node'] = np.append(total['gt_node'], gtnl)
    return reports, reports_txt


def handle_unpreds(gt: np.ndarray, hc: np.ndarray, drop: bool) -> Tuple[np.ndarray, np.ndarray]:
    if drop:
        mask = np.logical_and(hc != -1, gt != -1)
        return gt[mask], hc[mask]
    else:
        return gt, hc


def get_target_names(gtl: np.ndarray, hcl: np.ndarray, target_names: list) -> list:
    target_names = np.array(target_names)
    total = np.unique(np.concatenate((gtl, hcl), axis=0)).astype(int)
    return list(target_names[total])


def evaluate_validation_set(set_path: str, gt_path: str, total=True, direct: bool = False, filters: bool = False,
                            drop_unpreds: bool = True, data_type: str = 'ce'):
    set_path = os.path.expanduser(set_path)
    gt_path = os.path.expanduser(gt_path)
    dirs = os.listdir(set_path)
    today = date.today().strftime("%Y_%m_%d")
    target_names = ['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head']
    reports = {}
    for di in dirs:
        input_path = set_path + di + '/'
        report = eval_dataset(input_path, gt_path, input_path + 'evaluation/', total=total, direct=direct,
                              filters=filters, drop_unpreds=drop_unpreds, data_type=data_type,
                              report_name='eval_' + today, target_names=target_names)
        argscont = basics.load_pkl(input_path + 'info/argscont.pkl')
        report.update(argscont)
        reports[di] = report
    basics.save2pkl(reports, set_path + 'evaluation/', name='eval_' + today)




if __name__ == '__main__':
    evaluate_validation_set('~/thesis/trainings/past/param_search_2/validation/',
                            '~/thesis/gt/20_02_20/poisson/validation/', total=True)
