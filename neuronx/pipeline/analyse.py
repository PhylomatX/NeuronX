import os
import re
from morphx.processing import basics
import matplotlib.pyplot as plt
from typing import List


# -------------------------------------- REPORT HANDLING ------------------------------------------- #

def summarize_reports(set_path: str, eval_name: str):
    """
    Combines all reports from multiple evaluations (e.g. checkpoint evaluations) to a single report.
    """
    set_path = os.path.expanduser(set_path)
    dirs = os.listdir(set_path)
    reports = {}
    for di in dirs:
        input_path = set_path + di + '/'
        if not os.path.exists(input_path + f'{eval_name}/{eval_name}' + '.pkl'):
            continue
        report = basics.load_pkl(input_path + f'{eval_name}/{eval_name}' + '.pkl')
        argscont = basics.load_pkl(input_path + 'argscont.pkl')
        report.update(argscont)
        reports[di] = report
    basics.save2pkl(reports, set_path, name=eval_name)


def extract_data_from_reports(reports_path: str,
                              cell_key: str = 'total',
                              mode_key: str = 'mv',
                              class_key: str = 'macro avg',
                              metric_key: str = 'f1-score'):
    """
    Extracts the requested metric from the reports file at reports_path and creates a datacontainer
    which can then be transformed into a diagram.

    reports structure:
    1. directories (e.g. '2020_03_14_50_5000'),
    2. cell predictions, total predictions and argument container keywords  (e.g. 'total', 'sso_46313345_preds', 'sample_num', ...),
    3. evaluation mode (e.g. 'mv' for majority vote results on vertex level or 'mv_skel' for results on node level),
    4. classes and meta metric keys (e.g. 'dendrite', 'axon', 'accuracy', macro avg'),
    5. actual score (e.g. for 'accuracy') or metric keys (e.g. 'precision', 'recall', 'f1-score'),
    6. actual scores

    Args:
        reports_path: path to reports file.
        cell_key: Choose between single cells (e.g. 'sso_491527_poisson') or choose 'total'
        mode_key: Choose between mesh (e.g. 'mv') or skeleton (e.g. 'mv_skel') evaluation
        class_key: Chosse between classes (e.g. 'dendrite', 'axon', ...) or averages (e.g. 'accuracy', 'macro avg', ...)
        metric_key: Choose between metrics (e.g. 'precision', 'f1-score', ...)

    Return:
        Data container with requested metrics.
    """
    reports = basics.load_pkl(reports_path)
    epochs = []
    scores = []
    for key in list(reports.keys()):
        report = reports[key]
        reports.pop(key)
        metric = report[cell_key][mode_key][class_key]
        if class_key == 'accuracy':
            score = metric
        else:
            score = metric[metric_key]
        epoch_num = int(re.findall(r"epoch_(\d+)", key)[0])
        epochs.append(epoch_num)
        scores.append(score)
    return epochs, scores


# -------------------------------------- DIAGRAM GENERATION ------------------------------------------- #

def generate_diagram(reports_path: str,
                     output_path: str,
                     cell_key: str = 'total',
                     mode_key: str = 'mv',
                     class_key: str = 'macro avg',
                     metric_key: str = 'f1-score'):
    reports_path = os.path.expanduser(reports_path)
    output_path = os.path.expanduser(output_path)
    epochs, scores = extract_data_from_reports(reports_path, cell_key, mode_key, class_key, metric_key)
    fig, ax = plt.subplots()
    fontsize = 15
    ax.scatter(epochs, scores, zorder=3)
    ax.set_xlabel(f'epoch number', fontsize=fontsize, labelpad=10)
    ax.set_ylabel(metric_key, fontsize=fontsize, labelpad=10)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    plt.ylim((0, 1))
    plt.savefig(output_path + f"{cell_key}_{mode_key}_{class_key}_{metric_key}.png")


def generate_class_diagrams(reports_path: str,
                            output_path: str,
                            part_key: str,
                            class_keys: List[str],
                            metric_key: str = 'f1-score'):
    for class_key in class_keys:
        generate_diagram(reports_path, output_path, mode_key=part_key, class_key=class_key, metric_key=metric_key)  # vertex level
        generate_diagram(reports_path, output_path, mode_key=part_key + '_skel', class_key=class_key, metric_key=metric_key)  # skeleton level
    generate_diagram(reports_path, output_path, mode_key=part_key, class_key='accuracy')  # vertex level, accuracy
    generate_diagram(reports_path, output_path, mode_key=part_key + '_skel', class_key='accuracy')  # skeleton level, accuracy
