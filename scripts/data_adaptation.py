import os
import re
import glob
import numpy as np
from tqdm import tqdm
from morphx.data import basics
from syconn import global_params
from morphx.processing import ensembles
from morphx.preprocessing import mesh2poisson as m2p
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property


def transfer_skeleton(origin: str, target: str, output: str):
    files = glob.glob(target + '*.pkl')
    if not os.path.isdir(output):
        os.makedirs(output)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        hc = basics.load_pkl(origin + name + '_c.pkl')
        ce = ensembles.ensemble_from_pkl(file)
        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
        hc.set_encoding(encoding)
        ce.change_hybrid(hc)
        ce.save2pkl(output + name + '.pkl')


def generate_verts2node(input_path: str, output_path: str):
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    files = glob.glob(input_path + '*.pkl')
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        ce = ensembles.ensemble_from_pkl(file)
        ce.reset_ensemble()
        ce.hc.set_verts2node(None)
        ce.save2pkl(output_path + name + '.pkl')


def add_myelin(set_path: str, out_path: str):
    """ loads myelin predictions, maps them to node coordinates, then maps them to vertices and saves all
        CloudEnsembles at given path with the myelin predicitons included to out_path. """
    set_path = os.path.expanduser(set_path)
    out_path = os.path.expanduser(out_path)
    files = glob.glob(set_path + '*_poisson.pkl')
    finished = glob.glob(out_path + '*.pkl')
    for file in tqdm(files):
        sso_id = int(re.findall(r"/sso_(\d+).", file)[0])
        if sso_id == 4741011 or sso_id == 26331138:
            global_params.wd = "/wholebrain/scratch/areaxfs3/"
            version = 'spgt'
        else:
            global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
            version = None
        for item in finished:
            if sso_id in item:
                print(f"{file} has already been processed and gets skipped.")
                continue
        sso = SuperSegmentationObject(sso_id, version=version)
        sso.load_skeleton()
        sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
        majorityvote_skeleton_property(sso, 'myelin')
        myelinated = sso.skeleton['myelin_avg10000']
        ce = ensembles.ensemble_from_pkl(file)
        hc = ce.hc
        nodes_idcs = np.arange(len(hc.nodes))
        myel_nodes = nodes_idcs[myelinated.astype(bool)]
        myel_vertices = []
        for node in myel_nodes:
            myel_vertices.extend(hc.verts2node[node])
        # myelinated vertices get type 1, not myelinated vertices get type 0
        types = np.zeros(len(hc.vertices))
        types[myel_vertices] = 1
        hc.set_types(types)
        ce.save2pkl(out_path + f'sso_{sso_id}.pkl')


def poissonize(input_path: str, output_path: str, tech_density: int):
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    m2p.poissonize_dataset(input_path, output_path, tech_density)


if __name__ == '__main__':
    # add_myelin('~/thesis/gt/20_04_09/evaluation/', '~/thesis/gt/20_04_16/evaluation/')
    # poissonize('~/thesis/gt/intermediate/', '~/thesis/gt/intermediate/', 1500)
    # generate_verts2node('~/thesis/gt/intermediate/', '~/thesis/gt/intermediate/')
    add_myelin('~/thesis/gt/intermediate/', '~/thesis/gt/intermediate/myelin/')
