import os
import glob
import numpy as np
import sklearn.metrics as sm
from tqdm import tqdm
from morphx.processing import objects
from morphx.data import basics
import matplotlib.pyplot as plt
from typing import List, Tuple
from neuronx.classes.datacontainer import DataContainer
from neuronx.classes.argscontainer import ArgsContainer
from neuronx.pipeline import validate as val


# -------------------------------------- EVALUATION METHODS ------------------------------------------- #

def eval_dataset(input_path: str, output_path: str, argscont: ArgsContainer, report_name: str = 'Evaluation',
                 total: bool = False, mode: str = 'mvs', filters: bool = False, drop_unpreds: bool = True,
                 data_type: str = 'ce', target_names: list = None):
    """ Apply different metrics to HybridClouds with predictions and compare these predictions with corresponding
        ground truth files with different filters or under different conditions.

    Args:
        input_path: Location of HybridClouds with predictions, saved as pickle files by a MorphX prediction mapper.
        output_path: Location where results of evaluation should be saved.
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
        target_names: encoding of labels
    """
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(input_path + 'sso_*.pkl')
    reports = {}
    reports_txt = ""
    # Arrays for concatenating the labels of all files for later total evaluation
    total_labels = {'pred': np.array([]), 'pred_node': np.array([]), 'gt': np.array([]), 'gt_node': np.array([])}
    # Build single file evaluation reports
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        report, report_txt = eval_single(file, total_labels, mode=mode, target_names=target_names, filters=filters,
                                         drop_unpreds=drop_unpreds, data_type=data_type,
                                         label_mapping=argscont.label_mappings)
        reports[name] = report
        reports_txt += name + '\n\n' + report_txt + '\n\n\n'
    # Perform evaluation on total label arrays (labels from all files sticked together), prediction
    # mappings or filters are already included
    total_report = {}
    total_report_txt = 'Total\n\n'
    if total:
        targets = get_target_names(total_labels['gt'], total_labels['pred'], target_names)
        total_report[mode] = \
            sm.classification_report(total_labels['gt'], total_labels['pred'], output_dict=True, target_names=targets)
        total_report_txt += \
            mode + '\n\n' + \
            sm.classification_report(total_labels['gt'], total_labels['pred'], target_names=targets) + '\n\n'
        mode += '_skel'
        if filters:
            mode += '_f'
        targets = get_target_names(total_labels['gt_node'], total_labels['pred_node'], target_names)
        total_report[mode] = \
            sm.classification_report(total_labels['gt_node'], total_labels['pred_node'],
                                     output_dict=True, target_names=targets)
        total_report_txt += \
            mode + '\n\n' + sm.classification_report(total_labels['gt_node'], total_labels['pred_node'],
                                                     target_names=targets) + '\n\n'
    reports['total'] = total_report
    reports_txt += total_report_txt
    basics.save2pkl(reports, output_path, name=report_name)
    with open(output_path + report_name + '.txt', 'w') as f:
        f.write(reports_txt)
    return reports


def eval_single(file: str, total: dict = None, mode: str = 'mvs', target_names: list = None, filters: bool = False,
                drop_unpreds: bool = True, data_type: str = 'obj', label_mapping: List[Tuple[int, int]] = None) -> tuple:
    """ Apply different metrics to HybridClouds with predictions and compare these predictions with corresponding
        ground truth files with different filters or under different conditions. See eval_dataset for argument
        description.

    Returns:
        Evaluation report as string.
    """
    file = os.path.expanduser(file)
    preds = basics.load_pkl(file)
    # load HybridCloud and corresponding ground truth
    obj = objects.load_obj(data_type, preds[0])
    obj.set_predictions(preds[1])
    if label_mapping is not None:
        obj.hc.map_labels([(3, 1), (4, 1), (5, 0), (6, 0)])
    reports = {}
    reports_txt = ""
    # Perform majority vote on existing predictions and set these as new labels
    if mode == 'd':
        obj.generate_pred_labels(False)
    elif mode == 'mv':
        obj.generate_pred_labels()
    elif mode == 'mvs':
        obj.generate_pred_labels()
        obj.hc.prediction_smoothing()
    else:
        raise ValueError(f"Mode {mode} is not known.")
    hc = obj.hc
    if len(hc.pred_labels) != len(hc.labels):
        raise ValueError("Length of predicted label array doesn't match with length of label array.")
    # Get evaluation for vertices
    gtl, hcl = handle_unpreds(hc.labels, hc.pred_labels, drop_unpreds)
    targets = get_target_names(gtl, hcl, target_names)
    reports[mode] = sm.classification_report(gtl, hcl, output_dict=True, target_names=targets)
    reports_txt += mode + '\n\n' + sm.classification_report(gtl, hcl, target_names=targets) + '\n\n'
    # Get evaluation for skeletons
    mode += '_skel'
    if filters:
        hc.clean_node_labels()
        mode += '_f'
    gtnl, hcnl = handle_unpreds(hc.node_labels, hc.pred_node_labels, drop_unpreds)
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


def evaluate_validation_set(set_path: str, total=True, mode: str = 'mvs', filters: bool = False,
                            drop_unpreds: bool = True, data_type: str = 'ce', eval_name: str = 'evaluation'):
    """ Evaluates validations from multiple trainings.

    Args:
        set_path: path to validation folders
        total: flag for generating a total evaluation
        mode: 'd': direct mode (first prediction is taken), 'mv': majority vote mode (majority vote on predictions)
            'mvs' majority vote smoothing mode (majority vote on predicitons and smoothing afterwards)
        filters: flag for applying filters to skeleton predictions
        drop_unpreds: flag for removing vertices without predictions
        data_type: type of dataset ('ce' for CloudEnsembles, 'hc' for HybridClouds)
        eval_name: name of evaluation
    """
    set_path = os.path.expanduser(set_path)
    dirs = os.listdir(set_path)
    target_names = ['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head']
    reports = {}
    for di in tqdm(dirs):
        di_in_path = set_path + di + '/'
        di_out_path = set_path + di + '/' + eval_name + '/'
        if os.path.exists(di_out_path):
            print(di + " has already been processed. Skipping...")
            continue
        print("Processing " + di)
        if not os.path.exists(di_in_path + 'argscont.pkl'):
            print(f'{di} has no argscont.pkl file and gets skipped...')
            continue
        argscont = ArgsContainer().load_from_pkl(di_in_path + 'argscont.pkl')
        report = eval_dataset(di_in_path, di_out_path, argscont, report_name=eval_name, total=total, mode=mode,
                              filters=filters, drop_unpreds=drop_unpreds, data_type=data_type,
                              target_names=target_names)
        report.update(argscont.attr_dict)
        reports[di] = report
    basics.save2pkl(reports, set_path, name=eval_name)


# -------------------------------------- HELPER METHODS ------------------------------------------- #

def handle_unpreds(gt: np.ndarray, hc: np.ndarray, drop: bool) -> Tuple[np.ndarray, np.ndarray]:
    """ Removes labels which equal -1. """
    if drop:
        mask = np.logical_and(hc != -1, gt != -1)
        return gt[mask], hc[mask]
    else:
        return gt, hc


def get_target_names(gtl: np.ndarray, hcl: np.ndarray, target_names: list) -> list:
    """ Extracts the names of the labels which appear in gtl and hcl. """
    target_names = np.array(target_names)
    total = np.unique(np.concatenate((gtl, hcl), axis=0)).astype(int)
    return list(target_names[total])


# -------------------------------------- REPORT HANDLING ------------------------------------------- #

def summarize_reports(set_path: str, eval_name: str):
    """ Combines all reports on individual training level to a single report on training set level. """
    set_path = os.path.expanduser(set_path)
    dirs = os.listdir(set_path)
    reports = {}
    for di in dirs:
        input_path = set_path + di + '/'
        if not os.path.exists(input_path + f'{eval_name}/{eval_name}' + '.pkl'):
            print(f'{di} has no evaluation file and gets skipped...')
            continue
        report = basics.load_pkl(input_path + f'{eval_name}/{eval_name}' + '.pkl')
        argscont = basics.load_pkl(input_path + 'argscont.pkl')
        report.update(argscont)
        reports[di] = report
    basics.save2pkl(reports, set_path, name=eval_name)


# -------------------------------------- PIPELINE METHODS ------------------------------------------- #

def full_evaluation_pipe(set_path: str, val_path, out_path, total=True, mode: str = 'mv', filters: bool = False,
                         drop_unpreds: bool = True, data_type: str = 'ce', cell_key: str = 'total',
                         part_key: str = 'mv', class_key: str = 'macro avg', metric_key: str = 'f1-score',
                         eval_name: str = 'evaluation', pipe_steps=None, val_iter=2):
    """ Runs full pipeline on given training set (including validation, evaluation and diagram generation. """
    if pipe_steps is None:
        pipe_steps = [True, True]
    out_path += set_path + f'evaluation_valiter{val_iter}'
    eval_name += f'_{mode}'
    set_path = os.path.expanduser(set_path)
    val_path = os.path.expanduser(val_path)
    out_path = os.path.expanduser(out_path)
    if pipe_steps[0]:
        # run validations
        val.validate_training_set(set_path, val_path, out_path, model_type='state_dict_final.pth', val_iter=val_iter)
    if pipe_steps[1]:
        # evaluate validations
        evaluate_validation_set(out_path, total, mode, filters, drop_unpreds, data_type, eval_name=eval_name)


# -------------------------------------- DIAGRAM GENERATION ------------------------------------------- #

def reports2data(reports_path: str, identifier: List[str], cell_key: str = 'total', part_key: str = 'mv',
                 class_key: str = 'macro avg', metric_key: str = 'f1-score', points: bool = False):
    """ Extracts the requested metric from the reports file at reports_path and creates a datacontainer
        which can then be transformed into a diagram.

        reports structure keys:
        1. directories (e.g. '2020_03_14_50_5000'),
        2. sso preds and argscont (e.g. 'sso_46313345_preds', 'sample_num', 'density_mode'),
        3. sso structures (e.g. 'mv' or 'mv_skel'),
        4. classes (e.g. 'dendrite', 'axon', 'accuracy', macro avg'),
        5. int (e.g. for 'accuracy') or metrics (e.g. 'precision', 'recall', 'f1-score'),
        6. ints

    Args:
        reports_path: path to reports file.
        identifier: list of strings with identifiers which can be used to seperate different training modes
        cell_key: Choose between single cells (e.g. 'sso_491527_poisson') or choose 'total'
        part_key: Choose between mesh (e.g. 'mv') or skeleton (e.g. 'mv_skel')
        class_key: Chosse between classes (e.g. 'dendrite', 'axon', ...) or averages (e.g. 'accuracy', 'macro avg', ...)
        metric_key: Choose between metrics (e.g. 'precision', 'f1-score', ...)
        points: flag for saving points with f1-score, keyed by density or chunk size
    """
    reports = basics.load_pkl(reports_path)
    dataconts = []
    for ix in range(len(identifier)+1):
        keys = []
        for key in reports.keys():
            if ix >= len(identifier):
                keys.append(key)
            else:
                if identifier[ix] in key:
                    keys.append(key)
        density_data = {}
        context_data = {}
        for key in keys:
            report = reports[key]
            reports.pop(key)
            density_mode = report['density_mode']
            metric = report[cell_key][part_key][class_key]
            if not isinstance(metric, int) and not isinstance(metric, float):
                metric = metric[metric_key]
            sample_num = report['sample_num']
            if points:
                if density_mode:
                    bio_density = report['bio_density']
                    if bio_density in density_data.keys():
                        density_data[bio_density][0].append(sample_num)
                        density_data[bio_density][1].append(metric)
                    else:
                        density_data[bio_density] = ([sample_num], [metric])
                else:
                    chunk_size = report['chunk_size']
                    if chunk_size in context_data.keys():
                        context_data[chunk_size][0].append(sample_num)
                        context_data[chunk_size][1].append(metric)
                    else:
                        context_data[chunk_size] = ([sample_num], [metric])
            else:
                if density_mode:
                    bio_density = report['bio_density']
                    if sample_num in density_data.keys():
                        density_data[sample_num][0].append(bio_density)
                        density_data[sample_num][1].append(metric)
                    else:
                        density_data[sample_num] = ([bio_density], [metric])
                else:
                    chunk_size = report['chunk_size']
                    if sample_num in context_data.keys():
                        context_data[sample_num][0].append(chunk_size)
                        context_data[sample_num][1].append(metric)
                    else:
                        context_data[sample_num] = ([chunk_size], [metric])
        if class_key == 'accuracy':
            metric = 'accuracy'
        else:
            metric = metric_key
        datacont = DataContainer(density_data, context_data, metric=metric)
        dataconts.append(datacont)
    return dataconts


def generate_diagram(reports_path: str, output_path: str, identifier: List[str], ident_labels: List[str],
                     cell_key: str = 'total', part_key: str = 'mv', class_key: str = 'macro avg',
                     metric_key: str = 'f1-score', density: bool = True, points: bool = False):
    """ Generates diagram which visualizes the parameter search.
        data structure:
        density_data: Tuple of lists, 0: list of densities, 1: list of metrics, keyed by the sample number
        context_data: Tuple of lists, 0: list of chunk sizes 1: list of metrics, keyed by the sample number
    """
    reports_path = os.path.expanduser(reports_path)
    output_path = os.path.expanduser(output_path)
    dataconts = reports2data(reports_path, identifier, cell_key, part_key, class_key, metric_key, points=points)
    fig, ax = plt.subplots()
    markers = ['o', 'x', '+', 'S', '^', 'H']
    for data_ix, data in enumerate(dataconts):
        density_data = data.density_data
        context_data = data.context_data
        colors = ['b', 'k', 'g', 'r', 'c',  'y', 'm']
        if density and not points:
            for ix, key in enumerate(density_data.keys()):
                densities = density_data[key][0]
                metrics = density_data[key][1]
                ax.scatter(densities, metrics, c=colors[ix], label=ident_labels[data_ix] + '_' + str(key),
                           marker=markers[data_ix])
                ax.set_xlabel(r'points/\microm²')
                ax.set_ylabel(data.metric)
        elif not density and not points:
            for ix, key in enumerate(context_data.keys()):
                contexts = context_data[key][0]
                metrics = context_data[key][1]
                ax.scatter(contexts, metrics, c=colors[ix], label=ident_labels[data_ix] + '_' + str(key),
                           marker=markers[data_ix])
                ax.set_xlabel(r'context size in \nanom')
                ax.set_ylabel(data.metric)
        elif density and points:
            for ix, key in enumerate(density_data.keys()):
                point_nums = density_data[key][0]
                metrics = density_data[key][1]
                ax.scatter(point_nums, metrics, c=colors[ix], label=ident_labels[data_ix] + '_' + str(key),
                           marker=markers[data_ix])
                ax.set_xlabel('number of points')
                ax.set_ylabel(data.metric)
        elif not density and points:
            for ix, key in enumerate(context_data.keys()):
                point_nums = context_data[key][0]
                metrics = context_data[key][1]
                ax.scatter(point_nums, metrics, c=colors[ix], label=ident_labels[data_ix] + '_' + str(key),
                           marker=markers[data_ix])
                ax.set_xlabel('number of points')
                ax.set_ylabel(data.metric)
    ax.legend(loc=0)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(output_path + f"{cell_key}_{part_key}_{class_key}_{metric_key}_d{density}_p{points}.png")


if __name__ == '__main__':
    # s_path = '~/thesis/trainings/past/param_search_density/'
    # v_path = '~/thesis/gt/20_02_20/poisson_verts2node/validation/'
    # full_evaluation_pipe(s_path, v_path, pipe_steps=[False, True, False])

    r_path = '~/thesis/trainings/past/param_search_density/evaluation_valiter2/evaluation_mvs.pkl'
    o_path = '~/thesis/trainings/past/param_search_density/evaluation_valiter2/'
    generate_diagram(r_path, o_path, ['co'], ['cell organelles', 'no cell organelles'], points=True,
                     part_key='mvs', class_key='accuracy')
