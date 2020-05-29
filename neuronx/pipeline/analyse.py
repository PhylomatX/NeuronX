import os
import re
import numpy as np
from morphx.data import basics
import matplotlib.pyplot as plt
from typing import List
from neuronx.classes.datacontainer import DataContainer


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


def reports2data(reports_path: str, identifier: List[str], cell_key: str = 'total', part_key: str = 'mv',
                 class_key: str = 'macro avg', metric_key: str = 'f1-score', points: bool = False,
                 filter_identifier: bool = False, neg_identifier: List[str] = None, time: bool = False):
    """ Extracts the requested metric from the reports file at reports_path and creates a datacontainer
        which can then be transformed into a diagram.

        reports structure keys:
        1. directories (e.g. '2020_03_14_50_5000'),
        2. sso preds and argscont (e.g. 'sso_46313345_preds', 'sample_num', 'density_mode', ...),
        3. sso structures (e.g. 'mv' or 'mv_skel'),
        4. classes (e.g. 'dendrite', 'axon', 'accuracy', macro avg'),
        5. actual score (e.g. for 'accuracy') or metrics (e.g. 'precision', 'recall', 'f1-score'),
        6. actual score

    Args:
        reports_path: path to reports file.
        identifier: list of strings with identifiers which can be used to seperate different training modes
        cell_key: Choose between single cells (e.g. 'sso_491527_poisson') or choose 'total'
        part_key: Choose between mesh (e.g. 'mv') or skeleton (e.g. 'mv_skel')
        class_key: Chosse between classes (e.g. 'dendrite', 'axon', ...) or averages (e.g. 'accuracy', 'macro avg', ...)
        metric_key: Choose between metrics (e.g. 'precision', 'f1-score', ...)
        points: flag for saving points with metric, keyed by density or chunk size
        time: flag for saving epochs with metric, keyed by density or chunk size
        filter_identifier: only include reports whose keys include the given identifiers into the data container
        neg_identifier: exclude reports whose keys include the given negative identifier from the data container

    Return:
        Data container with tuples like (param1, metric), keyed by param2, where param1 and param2 can be:
        'bio_density', 'sample_num', 'chunk_size', ...
    """
    reports = basics.load_pkl(reports_path)
    dataconts = []
    # filter or exclude reports by given identifier criteria
    for ix in range(len(identifier)+1):
        keys = []
        for key in reports.keys():
            if ix >= len(identifier):
                if not filter_identifier:
                    keys.append(key)
            else:
                if identifier[ix] in key:
                    valid = True
                    for neg_ident in neg_identifier:
                        if neg_ident in key:
                            valid = False
                    if valid:
                        keys.append(key)
        # prepare data
        density_data = {}
        context_data = {}
        time_data = {}
        for key in keys:
            report = reports[key]
            reports.pop(key)
            density_mode = report['density_mode']
            # handle 'accuracy' vs. other metrics
            metric = report[cell_key][part_key][class_key]
            if not isinstance(metric, int) and not isinstance(metric, float):
                metric = metric[metric_key]
            sample_num = report['sample_num']
            if time:
                epoch_num = int(re.findall(r"epoch_(\d+)", key)[0])
                if density_mode:
                    # ([epoch_nums], [metrics]) keyed by density
                    param = report['bio_density']
                else:
                    # ([epoch_nums], [metrics]) keyed by context
                    param = report['chunk_size']
                if param in time_data.keys():
                    time_data[param][0].append(epoch_num)
                    time_data[param][1].append(metric)
                else:
                    time_data[param] = ([epoch_num], [metric])
            elif points:
                if density_mode:
                    # ([sample_nums], [metrics]) keyed by density
                    bio_density = report['bio_density']
                    if bio_density in density_data.keys():
                        density_data[bio_density][0].append(sample_num)
                        density_data[bio_density][1].append(metric)
                    else:
                        density_data[bio_density] = ([sample_num], [metric])
                else:
                    # ([sample_nums], [metrics]) keyed by context
                    chunk_size = report['chunk_size']
                    if chunk_size in context_data.keys():
                        context_data[chunk_size][0].append(sample_num)
                        context_data[chunk_size][1].append(metric)
                    else:
                        context_data[chunk_size] = ([sample_num], [metric])
            else:
                if density_mode:
                    # ([densities], [metrics]) keyed by sample_num
                    bio_density = report['bio_density']
                    if sample_num in density_data.keys():
                        density_data[sample_num][0].append(bio_density)
                        density_data[sample_num][1].append(metric)
                    else:
                        density_data[sample_num] = ([bio_density], [metric])
                else:
                    # ([contexts], [metrics]) keyed by sample_num
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
        datacont = DataContainer(density_data, context_data, time_data, metric=metric)
        dataconts.append(datacont)
    return dataconts


# -------------------------------------- DIAGRAM GENERATION ------------------------------------------- #

def generate_diagram(reports_path: str, output_path: str, identifier: List[str], ident_labels: List[str],
                     cell_key: str = 'total', part_key: str = 'mv', class_key: str = 'macro avg',
                     metric_key: str = 'f1-score', density: bool = True, points: bool = False, time: bool = False,
                     filter_identifier: bool = False, neg_identifier: List[str] = None):
    """ Generates diagram which visualizes the parameter search.
        data structure:
        density_data: Tuple of lists, 0: list of densities, 1: list of metrics, keyed by sample number / density
        context_data: Tuple of lists, 0: list of chunk sizes 1: list of metrics, keyed by sample number / context
        context_data: Tuple of lists, 0: list of epochs 1: list of metrics, keyed by density / context
    """
    reports_path = os.path.expanduser(reports_path)
    output_path = os.path.expanduser(output_path)
    dataconts = reports2data(reports_path, identifier, cell_key, part_key, class_key, metric_key, points=points,
                             filter_identifier=filter_identifier, neg_identifier=neg_identifier, time=time)
    fig, ax = plt.subplots()
    markers = ['o', 'x', '+', 'S', '^', 'H']
    fontsize = 15
    for data_ix, data in enumerate(dataconts):
        density_data = data.density_data
        context_data = data.context_data
        time_data = data.time_data
        colors = ['b', 'k', 'g', 'r', 'c',  'y', 'm']
        # time on x-axis, metrics on y-axis, different contexts / densities indicated by colors
        if time:
            for ix, key in enumerate(time_data.keys()):
                times = time_data[key][0]
                metrics = time_data[key][1]
                if density:
                    param = '1/\u03BCm^2'
                else:
                    param = '\u03BCm'
                ax.scatter(times, metrics, marker=markers[data_ix], c=colors[ix], zorder=3,
                           label=ident_labels[data_ix] + param)
                ax.set_xlabel(f'epoch number', fontsize=fontsize, labelpad=10)
                ax.set_ylabel(data.metric, fontsize=fontsize, labelpad=10)
        # density on x-axis, metrics on y-axis, different sample nums indicated by colors
        elif density and not points:
            for ix, key in enumerate(density_data.keys()):
                densities = density_data[key][0]
                metrics = density_data[key][1]
                ax.scatter(densities, metrics, marker=markers[data_ix], c=colors[ix], zorder=3,
                           label=ident_labels[data_ix] + f'{key} points')
                ax.set_xlabel(f'density in 1/\u03BC m²', fontsize=fontsize, labelpad=10)
                ax.set_ylabel(data.metric, fontsize=fontsize, labelpad=10)
        # context on x-axis, metrics on y-axis, different sample nums indicated by colors
        elif not density and not points:
            for ix, key in enumerate(context_data.keys()):
                contexts = np.array(context_data[key][0])/1000
                metrics = context_data[key][1]
                ax.scatter(contexts, metrics, marker=markers[data_ix], c=colors[ix], zorder=3,
                           label=ident_labels[data_ix] + f'{int(key/1e3)}k points')
                ax.set_xlabel('context size in \u03BCm', fontsize=fontsize, labelpad=10)
                ax.set_ylabel(data.metric, fontsize=fontsize, labelpad=10)
        # sample_num on x-axis, metrics on y-axis, different densities indicated by colors
        elif density and points:
            for ix, key in enumerate(density_data.keys()):
                point_nums = np.array(density_data[key][0])
                metrics = density_data[key][1]
                ax.scatter(np.round(point_nums/1e3), metrics, marker=markers[data_ix], c=colors[ix], zorder=3,
                           label=ident_labels[data_ix] + f'{key} p/\u03BCm²')
                ax.set_xlabel('number of points in 10³', fontsize=fontsize, labelpad=10)
                ax.set_ylabel(data.metric, fontsize=fontsize, labelpad=10)
        # context on x-axis, metrics on y-axis, different contexts indicated by colors
        elif not density and points:
            for ix, key in enumerate(context_data.keys()):
                point_nums = np.array(context_data[key][0])
                metrics = context_data[key][1]
                ax.scatter((point_nums/1e3), metrics, marker=markers[data_ix], c=colors[ix], zorder=3,
                           label=ident_labels[data_ix] + f'{int(key/1000)} \u03BCm')
                ax.set_xlabel('number of points in 10³', fontsize=fontsize, labelpad=10)
                ax.set_ylabel(data.metric, fontsize=fontsize, labelpad=10)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.legend(loc=0, fontsize=fontsize, ncol=1)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    plt.ylim(top=1)
    plt.savefig(output_path + f"{cell_key}_{part_key}_{class_key}_{metric_key}_d{density}_p{points}.png")


def generate_diagrams(reports_path: str, output_path: str, identifier: List[str], ident_labels: List[str],
                      points: bool, density: bool, part_key: str = 'mv', filter_identifier: bool = False,
                      neg_identifier: List[str] = None, time: bool = False):
    # vertex level, f1_score
    generate_diagram(reports_path, output_path, identifier, ident_labels, points=points, part_key=part_key,
                     density=density, filter_identifier=filter_identifier, neg_identifier=neg_identifier, time=time)
    # vertex level, accuracy
    generate_diagram(reports_path, output_path, identifier, ident_labels, points=points, part_key=part_key,
                     class_key='accuracy', density=density, filter_identifier=filter_identifier, time=time,
                     neg_identifier=neg_identifier)
    # skeleton level, f1_score
    generate_diagram(reports_path, output_path, identifier, ident_labels, points=points, part_key=part_key + '_skel',
                     density=density, filter_identifier=filter_identifier, neg_identifier=neg_identifier, time=time)
    # skeleton level, accuracy
    generate_diagram(reports_path, output_path, identifier, ident_labels, points=points, part_key=part_key + '_skel',
                     class_key='accuracy', density=density, filter_identifier=filter_identifier, time=time,
                     neg_identifier=neg_identifier)


if __name__ == '__main__':
    report_name = 'eval_mv'
    o_path = f'~/thesis/current_work/sp_3/run6/2020_05_26_2500_2000_16/eval_valiter3_batchsize-1/'
    summarize_reports(o_path, report_name)
    r_path = o_path + report_name + '.pkl'
    generate_diagrams(r_path, o_path, [''], [''], points=False, density=False, part_key='mv',
                      filter_identifier=False, neg_identifier=[], time=True)
