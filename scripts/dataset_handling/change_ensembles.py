import os
import glob
import numpy as np
from tqdm import tqdm
from morphx.processing import ensembles, clouds, objects
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.pointcloud import PointCloud


def transfer_poisson(poisson_path: str, target_path: str, output_path: str):
    files = glob.glob(target_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]

        hc = objects.load_pkl(poisson_path + name + '.pkl')
        ce = objects.load_pkl(file)
        cell = ce.get_cloud('cell')

        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
        cell = HybridCloud(cell.nodes, cell.edges, hc.vertices, labels=hc.labels, encoding=encoding)
        ce.add_cloud(cell, 'cell')
        objects.save2pkl(ce, output_path, name=name + '_c')


def cell2hc(data_path: str, output_path: str):
    data_path = os.path.expanduser(data_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(data_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-6]
        ce = objects.load_pkl(file)
        ce.change_hybrid(ce.get_cloud('cell'))
        ce.remove_cloud('cell')
        objects.save2pkl(ce, output_path, name=name)
        

def add_features(data_path: str, output_path: str):
    data_path = os.path.expanduser(data_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(data_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        ce = objects.load_pkl(file)
        hc = ce.hc
        hc = HybridCloud(hc.nodes, hc.edges, hc.vertices, labels=hc.labels,
                         features=np.ones((len(hc.vertices), 1)),
                         encoding={'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3,
                                   'terminal': 4, 'neck': 5, 'head': 6})

        factor = 2
        cloud_dict = {}
        for key in ce.clouds:
            cloud = ce.clouds[key]
            cloud = PointCloud(cloud.vertices, labels=cloud.labels,
                               features=np.ones((len(cloud.vertices), 1))*factor,
                               encoding={key: factor+5})
            cloud_dict[key] = cloud
            factor += 1

        ce = CloudEnsemble(cloud_dict, hybrid=hc, no_pred=['mi', 'vc', 'syn'])
        objects.save2pkl(ce, output_path, name=name)


def create_ads(data_path: str, output_path: str):
    data_path = os.path.expanduser(data_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(data_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        ce = objects.load_pkl(file)
        hc = ce.hc
        hc = clouds.map_labels(hc, ['bouton', 'terminal'], 'axon')
        hc = clouds.map_labels(hc, ['neck', 'head'], 'dendrite')
        ce.change_hybrid(hc)
        objects.save2pkl(ce, output_path, name=name)


def update_saving_standard(data_path: str, output_path: str):
    data_path = os.path.expanduser(data_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(data_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        ce = objects.load_pkl(file)
        ce.hc.set_features(np.ones((len(ce.hc.vertices), 1)))
        ce.save2pkl(output_path + name + 'pkl')


def transfer_skeleton(origin: str, target: str, output: str):
    files = glob.glob(target + '*.pkl')
    if not os.path.isdir(output):
        os.makedirs(output)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]

        hc = objects.load_pkl(origin + name + '_c.pkl')
        ce = objects.load_pkl(file)

        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
        hc.set_encoding(encoding)

        ce.change_hybrid(hc)
        objects.save2pkl(ce, output, name=name)


def generate_verts2node(input_path: str, output_path: str):
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(input_path + '*.pkl')
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        ce = ensembles.ensemble_from_pkl(file)
        ce.save2pkl(output_path + name + '.pkl')


if __name__ == '__main__':
    generate_verts2node('~/thesis/gt/20_02_20/poisson_verts2node/',
                        '~/thesis/gt/20_02_20/poisson_verts2node/')
