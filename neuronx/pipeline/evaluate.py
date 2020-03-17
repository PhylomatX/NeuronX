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

def eval_dataset(input_path: str, output_path: str, report_name: str = 'Evaluation', total: bool = False,
                 mode: str = 'mvs', filters: bool = False, drop_unpreds: bool = True, data_type: str = 'ce',
                 target_names: list = None):
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
    argscont = ArgsContainer().load_from_pkl(input_path + 'argscont.pkl')
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
        obj.prediction_smoothing()
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


def evaluate_validation_set(set_path: str, gt_path: str, total=True, mode: str = 'mvs',
                            filters: bool = False, drop_unpreds: bool = True, data_type: str = 'ce',
                            eval_name: str = 'evaluation'):
    """ Evaluates validations from multiple trainings.

    Args:
        set_path: path to validation folders
        gt_path: path where gt files corresponding to validation files can be found
        total: flag for generating a total evaluation
        mode: 'd': direct mode (first prediction is taken), 'mv': majority vote mode (majority vote on predictions)
            'mvs' majority vote smoothing mode (majority vote on predicitons and smoothing afterwards)
        filters: flag for applying filters to skeleton predictions
        drop_unpreds: flag for removing vertices without predictions
        data_type: type of dataset ('ce' for CloudEnsembles, 'hc' for HybridClouds)
        eval_name: name of evaluation
    """
    set_path = os.path.expanduser(set_path)
    gt_path = os.path.expanduser(gt_path)
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
        report = eval_dataset(di_in_path, di_out_path, report_name=eval_name, total=total, mode=mode, filters=filters,
                              drop_unpreds=drop_unpreds, data_type=data_type, target_names=target_names)
        argscont = basics.load_pkl(di_in_path + 'argscont.pkl')
        report.update(argscont)
        reports[di] = report
    basics.save2pkl(reports, set_path + eval_name + '/', name=eval_name)


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

def summarize_reports(set_path: str, eval_date: str):
    """ Combines all reports on individual training level to a single report on training set level. """
    set_path = os.path.expanduser(set_path)
    dirs = os.listdir(set_path)
    reports = {}
    for di in dirs:
        input_path = set_path + di + '/'
        report = basics.load_pkl(input_path + 'evaluation/eval_' + eval_date + '.pkl')
        argscont = basics.load_pkl(input_path + 'argscont.pkl')
        report.update(argscont)
        reports[di] = report
    basics.save2pkl(reports, set_path + 'evaluation/', name='eval_' + eval_date)


def reports2data(reports_path: str, output_path: str, cell_key: str = 'total', part_key: str = 'mv',
                 class_key: str = 'macro avg', metric_key: str = 'f1-score'):
    """ Extracts the requested metric from the reports file at reports_path and creates a datacontainer
        which can then be transformed into a diagram.

    Args:
        reports_path: path to reports file.
        output_path: pickle file in which datacontainer should get saved.
        cell_key: Choose between single cells (e.g. 'sso_491527_poisson') or choose 'total'
        part_key: Choose between mesh (e.g. 'mv') or skeleton (e.g. 'mv_skel')
        class_key: Chosse between classes (e.g. 'dendrite', 'axon', ...) or averages (e.g. 'accuracy', 'macro avg', ...)
        metric_key: Choose between metrics (e.g. 'precision', 'f1-score', ...)
    """
    reports = basics.load_pkl(reports_path)
    density_data = {}
    context_data = {}
    metric = ""
    for key in reports.keys():
        report = reports[key]
        density_mode = report['density_mode']
        sample_num = report['sample_num']
        metric = report[cell_key][part_key][class_key][metric_key]
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
    datacont = DataContainer(density_data, context_data, metric=metric)
    datacont.save2pkl(output_path)


# -------------------------------------- PIPELINE METHODS ------------------------------------------- #

def full_evaluation_pipe(set_path: str, val_path, out_path, total=True, mode: str = 'mv', filters: bool = False,
                         drop_unpreds: bool = True, data_type: str = 'ce', cell_key: str = 'total',
                         part_key: str = 'mv', class_key: str = 'macro avg', metric_key: str = 'f1-score',
                         eval_name: str = 'evaluation'):
    """ Runs full pipeline on given training set (including validation, evaluation and diagram generation.

    Args:
        set_path: path of training set with multiple training folders which contain the trained models and
            either argscont.pkl or training_args.pkl.
        val_path: path to cell pickle files which should get used for validation and evaluation.
        out_path: path where evaluation results should get saved.
        total:
        mode:
        filters:
        drop_unpreds:
        data_type:
        class_key:
        part_key:
        metric_key:
        cell_key:
        eval_name:
    """
    set_path = os.path.expanduser(set_path)
    val_path = os.path.expanduser(val_path)
    out_path = os.path.expanduser(out_path)
    # run validations
    # val.validate_training_set(set_path, val_path, out_path)
    # evaluate validations
    evaluate_validation_set(out_path, val_path, total, mode, filters, drop_unpreds, data_type,
                            eval_name=eval_name)
    # tranform reports to data
    # cell_key = 'total'
    # reports_path = out_path + eval_name + '/'
    # data_path = reports_path + f'{cell_key}_{part_key}_{class_key}_{metric_key}_data.pkl'
    # reports2data(reports_path + eval_name + '.pkl', data_path, cell_key, part_key, class_key, metric_key)
    # # generate diagrams
    # diagram_path = reports_path + f'{cell_key}_{part_key}_{class_key}_{metric_key}_diagram.png'
    # diagram_param_search(data_path, diagram_path)


# -------------------------------------- DIAGRAM GENERATION ------------------------------------------- #

def diagram_param_search(data_path: str, output_path: str):
    """ Generates diagram which visualizes the parameter search. """
    data = basics.load_pkl(data_path)
    density_data = data['density_data']
    context_data = data['context_data']

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('point number')
    ax1.set_ylabel('f1-score')
    # ax2 = ax1.twiny()
    # ax2.set_xlabel('context size (nm)')

    colors = ['b', 'r', 'g', 'b', 'y', 'c', 'm']
    labels = []

    points = []
    metrics = []
    for key in density_data.keys():
        points.append(key)
        labels.append(str(key))
        metrics.append(density_data[key][1])
    plot = ax1.scatter(points, np.array(metrics), c='k', marker='.')

    # for ix, key in enumerate(density_data):
    #     densities = np.array(density_data[key][0])
    #     metrics = np.array(density_data[key][1])
    #     plot = ax1.scatter(densities, metrics, c='b', marker='o')
    #     labels.append("density, " + str(key))
    #     plots.append(plot)
    #
    # for ix, key in enumerate(context_data):
    #     context_sizes = np.array(context_data[key][0])
    #     metrics = np.array(context_data[key][1])
    #     plot = ax2.scatter(context_sizes, metrics, c=colors[ix], marker='x')
    #     labels.append("context, " + str(key))
    #     plots.append(plot)

    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # plt.tight_layout()
    ax1.grid()
    # ax1.legend(labels, loc='lower center', ncol=int(len(labels)/2), title="mode, sample num")
    plt.savefig(output_path)


if __name__ == '__main__':
    s_path = '~/thesis/trainings/intermediate/'
    v_path = '~/thesis/gt/20_02_20/poisson_verts2node/validation/'
    o_path = s_path + 'evaluation/'
    full_evaluation_pipe(s_path, v_path, o_path)
