import os
import pickle
import glob
import numpy as np


def analyse_variance(path: str, epoch: str, labels: list = None):
    """ Returns the standard deviation of the score given by the sequence of labels of all the experiments in path.
        Each experiment must therefore have an evaluation file according to the NeuronX standard.

    Args:
        path: path to the experiments.
        labels: sequence of keys with which the score can be extracted from the evaluation file.

    Returns:
        The standard deviation of the experiments.
    """
    if labels is None:
        labels = ['mv', 'weighted avg', 'f1-score']
    result = {}
    path = os.path.expanduser(path)
    dirs = os.listdir(path)
    for di in dirs:
        if not os.path.isdir(path + di):
            continue
        file = glob.glob(path + di + '/*.pkl')[0]
        f = open(file, 'rb')
        info = pickle.load(f)[epoch]
        for key in info.keys():
            if 'pred' in key or key == 'total':
                single = info[key]
                for label in labels:
                    single = single[label]
                if key in result.keys():
                    result[key].append(single)
                else:
                    result[key] = [single]
    std = {}
    for key in result:
        result[key] = np.array(result[key])
        std[key] = result[key].std()
    return std, result


def variance_report(in_path: str, out_path: str, epoch: int):
    out_path = os.path.expanduser(out_path)
    report = ''
    for cell in analyse_variance(in_path, f'epoch_{epoch}')[0]:
        report += cell + '\n\n'
        for key in ['mv', 'mv_skel']:
            report += key + '\n\n'
            report += f"\n{f'score':<20}{'std':<20}{'rel (%)':<20}{'mean'}"
            for subkey in ['dendrite', 'neck', 'head', 'macro avg', 'weighted avg']:
                stds, result = analyse_variance(in_path, f'epoch_{epoch}', [key, subkey, 'f1-score'])
                report += f"\n{f'{subkey}:':<20}{round(stds[cell], 4):<20}{round(stds[cell]/np.mean(result[cell])*100, 2):<20}{np.round(np.mean(result[cell]), 4)}"
            stds, result = analyse_variance(in_path, f'epoch_{epoch}', [key, 'accuracy'])
            report += f"\n{'accuracy:':<20}{round(stds[cell], 4):<20}{round(stds[cell]/np.mean(result[cell])*100, 2):<20}{np.round(np.mean(result[cell]), 4)}"
            report += '\n\n\n'
        report += '\n\n'
    with open(out_path, 'w') as f:
        f.write(report)


if __name__ == '__main__':
    path = '~/thesis/current_work/sp_3/variance_analysis/2020_05_26_100_2000/red5/'
    # analyse_variance(path, 'epoch_101', ['mv', 'head', 'f1-score'])
    variance_report(path, path + 'std_report.txt', 221)