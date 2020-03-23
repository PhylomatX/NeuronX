import os
import re
import glob
import pickle
import numpy as np
from morphx.data import basics
from syconn import global_params
from morphx.processing import clouds
from morphx.processing import ensembles
from morphx.data.chunkhandler import ChunkHandler
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import map_myelin2coords


def add_myelin(set_path: str, out_path: str):
    """ loads myelin predictions, maps them to node coordinates, then maps them to vertices and saves all
        CloudEnsembles at given path with the myelin predicitons included to out_path. """
    set_path = os.path.expanduser(set_path)
    out_path = os.path.expanduser(out_path)
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    files = glob.glob(set_path + '.pkl')
    for file in files:
        sso_id = int(re.findall(r"/sso_(\d+).", file)[0])
        sso = SuperSegmentationObject(sso_id)
        sso.load_skeleton()
        myelinated = map_myelin2coords(sso.skeleton["nodes"], mag=4)
        ce = ensembles.ensemble_from_pkl(set_path + file)
        hc = ce.hc
        nodes_idcs = np.arange(len(hc.nodes))
        myel_nodes = nodes_idcs[myelinated]
        myel_vertices = []
        for node in myel_nodes:
            myel_vertices.extend(hc.verts2node[node])
        # myelinated vertices get type 1, not myelinated vertices get type 0
        types = np.zeros(len(hc.vertices))
        types[myel_vertices] = 1
        hc.set_types(types)
        ce.save2pkl(out_path + file)


def produce_chunks():
    """ Create and save all resulting chunks of an dataset. """
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    train_transforms = [clouds.RandomVariation((-1, 1)), clouds.RandomRotate(),
                        clouds.Normalization(10000), clouds.Center()]
    train_transforms = clouds.Compose(train_transforms)
    path = os.path.expanduser('~/thesis/gt/20_02_20/')
    ch = ChunkHandler(path + 'poisson_val/', sample_num = 5000,
                      density_mode = False, specific = True, chunk_size = 10000, obj_feats = features, transform=train_transforms)
    for item in ch.obj_names:
        for i in range(ch.get_obj_length(item)):
            sample, idcs = ch[(item, i)]
            with open(f'{path}/test/{item}_{i}.pkl', 'wb') as f:
                pickle.dump(sample, f)
            f.close()


def calculate_error(set_path: str, out_path: str, mode: str = 'mv'):
    """ This method is really ugly and was written in nocturnal insomnia. This can be done much more elegant, but
        was practical at the time and only for one-time use...

        There were 4 trainings for 3 different total stepsizes each. For all 3 groups, the error of different
        metrics gets calculated using the student-t distribution for 4 data points. Then the resulting errors of
        all 3 groups get averaged for the final error. """
    set_path = os.path.expanduser(set_path)
    out_path = os.path.expanduser(out_path)
    dirs = os.listdir(set_path)
    at80_t = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
              'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
    at100_t = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
               'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
    at32_t = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
              'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
    for di in dirs:
        at80 = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
                'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
        at100 = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
                 'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
        at32 = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
                'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
        data = basics.load_pkl(set_path + di + f'/{di[:-21]}_mv.pkl')
        pool = None
        for key in data:
            if 'at80' in key:
                pool = at80
            elif 'at100' in key:
                pool = at100
            elif 'at32' in key:
                pool = at32
            total = data[key]['total'][mode]
            pool['accuracy'].append(total['accuracy'])
            for score in pool['macro avg']:
                pool['macro avg'][score].append(total['macro avg'][score])
            for score in pool['weighted avg']:
                pool['weighted avg'][score].append(total['weighted avg'][score])
        import ipdb
        ipdb.set_trace()
        student_t = 0.6  # 4 measurements
        at80_t['accuracy'].append(student_t * np.std(at80['accuracy']))
        at100_t['accuracy'].append(student_t * np.std(at80['accuracy']))
        at32_t['accuracy'].append(student_t * np.std(at80['accuracy']))
        for key in at80:
            if key == 'accuracy':
                continue
            else:
                for score in at80[key]:
                    at80_t[key][score].append(student_t * np.std(at80[key][score]))
        for key in at100:
            if key == 'accuracy':
                continue
            else:
                for score in at100[key]:
                    at100_t[key][score].append(student_t * np.std(at100[key][score]))
        for key in at32:
            if key == 'accuracy':
                continue
            else:
                for score in at32[key]:
                    at32_t[key][score].append(student_t * np.std(at32[key][score]))

    # average over all 3 validations
    for key in at80_t:
        if key == 'accuracy':
            at80_t[key] = np.mean(at80_t[key])
        else:
            for score in at80_t[key]:
                at80_t[key][score] = np.mean(at80_t[key][score])
    for key in at100_t:
        if key == 'accuracy':
            at100_t[key] = np.mean(at100_t[key])
        else:
            for score in at100_t[key]:
                at100_t[key][score] = np.mean(at100_t[key][score])
    for key in at32_t:
        if key == 'accuracy':
            at32_t[key] = np.mean(at32_t[key])
        else:
            for score in at32_t[key]:
                at32_t[key][score] = np.mean(at32_t[key][score])

    with open(out_path + 'error.txt', 'w') as f:
        f.write(f"Accuracy error: {(at80_t['accuracy'] + at100_t['accuracy'] + at32_t['accuracy']) / 3}\n")
        f.write("\nmacro avg:\n")
        f.write(f"Precision error: {(at80_t['macro avg']['precision'] + at100_t['macro avg']['precision'] + at32_t['macro avg']['precision']) / 3}\n")
        f.write(f"Recall error: {(at80_t['macro avg']['recall'] + at100_t['macro avg']['recall'] + at32_t['macro avg']['recall']) / 3}\n")
        f.write(f"f1 error: {(at80_t['macro avg']['f1-score'] + at100_t['macro avg']['f1-score'] + at32_t['macro avg']['f1-score']) / 3}\n")
        f.write("\nweighted avg:\n")
        f.write(f"Precision error: {(at80_t['weighted avg']['precision'] + at100_t['weighted avg']['precision'] + at32_t['weighted avg']['precision']) / 3}\n")
        f.write(f"Recall error: {(at80_t['weighted avg']['recall'] +at100_t['weighted avg']['recall'] + at32_t['weighted avg']['recall']) / 3}\n")
        f.write(f"f1 error: {(at80_t['weighted avg']['f1-score'] + at100_t['weighted avg']['f1-score'] + at32_t['weighted avg']['f1-score']) / 3}\n")
    f.close()
