import os
import glob
import ipdb
from morphx.processing import ensembles, clouds
from morphx.classes.hybridcloud import HybridCloud


def transfer_poisson(poisson_path: str, target_path: str, output_path: str):
    files = glob.glob(target_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]

        hc = clouds.load_cloud(poisson_path + name + '.pkl')
        ce = ensembles.load_ensemble(file)
        cell = ce.get_cloud('cell')

        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
        cell = HybridCloud(cell.nodes, cell.edges, hc.vertices, labels=hc.labels, encoding=encoding)
        ce.add_cloud(cell, 'cell')
        ensembles.save_ensemble(ce, output_path, name=name + '_c')


def cell2hc(data_path: str, output_path: str):
    data_path = os.path.expanduser(data_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(data_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-6]
        ce = ensembles.load_ensemble(file)
        ce.change_hybrid(ce.get_cloud('cell'))
        ce.remove_cloud('cell')
        ensembles.save_ensemble(ce, output_path, name=name)


def transfer_skeleton(origin: str, target: str, output: str):
    files = glob.glob(target + '*.pkl')
    if not os.path.isdir(output):
        os.makedirs(output)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]

        hc = clouds.load_cloud(origin + name + '_c.pkl')
        ce = ensembles.load_ensemble(file)

        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
        hc.change_encoding(encoding)

        ce.change_hybrid(hc)
        ensembles.save_ensemble(ce, output, name=name)


if __name__ == '__main__':
    # transfer_poisson('/u/jklimesch/gt/gt_poisson/',
    #                  '/u/jklimesch/gt/gt_ensembles/raw/',
    #                  '/u/jklimesch/gt/gt_ensembles/')
    # cell2hc('/u/jklimesch/thesis/gt/gt_ensembles/', '/u/jklimesch/thesis/gt/gt_ensembles/')
    transfer_skeleton('/u/jklimesch/thesis/gt/gt_poisson/', '/u/jklimesch/thesis/gt/gt_ensembles/',
                      '/u/jklimesch/thesis/gt/gt_ensembles/')
