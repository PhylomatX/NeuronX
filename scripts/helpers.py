import os
import re
import glob
import pickle
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
from morphx.data import basics
from syconn import global_params
from morphx.processing import clouds
from matplotlib import pyplot as plt
from morphx.processing import ensembles
from morphx.data.chunkhandler import ChunkHandler
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import map_myelin2coords


def get_areas(set_path: str, out_path: str):
    set_path = os.path.expanduser(set_path)
    out_path = os.path.expanduser(out_path)
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    files = glob.glob(set_path + '*.pkl')
    for file in tqdm(files):
        sso_id = int(re.findall(r"/sso_(\d+).", file)[0])
        sso = SuperSegmentationObject(sso_id)
        svs = sso.svs
        total_area = 0
        for obj in svs:
            total_area += obj.mesh_area
        info = f'{file}: {total_area}\n'
        with open(out_path, 'a') as f:
            f.write(info)
        f.close()


def produce_set_info():
    ch = ChunkHandler('/u/jklimesch/thesis/gt/20_02_20/poisson_val_my/validation/', sample_num=28000,
                      density_mode=True, bio_density=50, tech_density=1500, specific=True)
    info = ch.get_set_info()
    with open('/u/jklimesch/thesis/results/gt/eval_info.pkl', 'wb') as f:
        pickle.dump(info, f)


def eval_set_info(info_path: str, out_path: str, name: str):
    info_path = os.path.expanduser(info_path)
    info = basics.load_pkl(info_path)
    tcell_area = 0
    tmi_area = 0
    tmi_num = 0
    tsj_area = 0
    tsj_num = 0
    tvc_area = 0
    tvc_num = 0
    classes = {0: 'dendrite', 1: 'axon', 2: 'soma', 3: 'bouton', 4: 'terminal', 5: 'neck', 6: 'head'}
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    tspecs = {}
    for key in tqdm(info):
        if 'sso' in key:
            specs = {}
            info_txt = ""
            info_txt += '\n\n' + key
            sso_id = int(re.findall(r"sso_(\d+).", key)[0])
            sso = SuperSegmentationObject(sso_id)

            # calculate cell surface area
            svs = sso.svs
            cell_area = 0
            for obj in svs:
                cell_area += obj.mesh_area
            tcell_area += cell_area
            info_txt += f'\ncell area: {round(cell_area)}'
            specs['cell_area'] = cell_area

            # summarize mitos
            mis = sso.mis
            mi_area = 0
            for obj in mis:
                mi_area += obj.mesh_area
            mi_num = len(mis)
            tmi_area += mi_area
            tmi_num += mi_num
            info_txt += f'\nmi area: {round(mi_area)}' \
                        f'\nmi num: {mi_num}'
            specs['mi_area'] = mi_area
            specs['mi_num'] = mi_num

            # summarize syns
            sjs = sso.syn_ssv
            sj_area = 0
            for obj in sjs:
                sj_area += obj.mesh_area
            sj_num = len(sjs)
            tsj_area += sj_area
            tsj_num += sj_num
            info_txt += f'\nsj area: {round(sj_area)}' \
                        f'\nsj num: {sj_num}'
            specs['sj_area'] = sj_area
            specs['sj_num'] = sj_num

            # summarize vesicles
            vcs = sso.vcs
            vc_area = 0
            for obj in vcs:
                vc_area += obj.mesh_area
            vc_num = len(vcs)
            tvc_area += vc_area
            tvc_num += vc_num
            info_txt += f'\nvc area: {round(vc_area)}' \
                        f'\nvc num: {vc_num}'
            specs['vc_area'] = vc_area
            specs['vc_num'] = vc_num

            labels = info[key]['labels']
            cell_mask = labels[0] < 7
            orga_mask = labels[0] > 6
            cell_vertex_num = labels[1][cell_mask.reshape(-1)].sum()
            orga_vertex_num = labels[1][orga_mask.reshape(-1)].sum()
            assert cell_vertex_num + orga_vertex_num == info[key]['vertex_num']

            if 1 in info[key]['types'][0]:
                myelin = info[key]['types'][1][1] / cell_vertex_num
            else:
                myelin = 0
            info_txt += f'\nmyelin coverage: {round(myelin*100, 1)}'
            specs['my'] = myelin

            for ix, label in enumerate(labels[0]):
                if label < 7:
                    cov = labels[1][ix] / cell_vertex_num
                    info_txt += f'\n{classes[label]} ({label}) percentage: {round(cov*100, 1)}'
                    specs[label] = cov

            info_txt += f"\ntotal vertex number: {info[key]['vertex_num']}"
            info_txt += f"\ncell vertex number: {cell_vertex_num}"
            info_txt += f"\nnode number: {info[key]['node_num']}"
            specs['vertex_num'] = info[key]['vertex_num']
            specs['cell_vertex_num'] = cell_vertex_num
            specs['node_num'] = info[key]['node_num']
            tspecs[key] = specs

            for i in range(7):
                try:
                    _ = specs[i]
                except KeyError:
                    specs[i] = 0.0

            info_txt += f"\n{sso_id} & {round(cell_area)} & {info[key]['node_num']} & " \
                        f"{round(myelin*100, 1)} & {round(specs[0]*100, 1)} & {round(specs[1]*100, 1)} & " \
                        f"{round(specs[2]*100, 1)} & {round(specs[3]*100, 1)} & {round(specs[4]*100, 1)} & " \
                        f"{round(specs[5]*100, 1)} & {round(specs[6]*100, 1)} \\"

            info_txt += f"\n{sso_id} & {round(mi_area)} ({mi_num}) & {round(sj_area)} ({sj_num}) & " \
                        f"{round(vc_area)} ({vc_num}) \\"

            with open(out_path + name + '.txt', 'a') as f:
                f.write(info_txt)
                f.close()

    info_txt = ""
    info_txt += '\n\ntotal:'
    info_txt += f'\ncell area: {round(tcell_area)}'
    info_txt += f'\nmi area: {round(tmi_area)}'
    info_txt += f'\nmi num: {tmi_num}'
    info_txt += f'\nsj area: {round(tsj_area)}'
    info_txt += f'\nsj num: {tsj_num}'
    info_txt += f'\nvc area: {round(tvc_area)}'
    info_txt += f'\nvc num: {tvc_num}'
    tspecs['cell_area'] = tcell_area
    tspecs['mi_area'] = tmi_area
    tspecs['sj_area'] = tsj_area
    tspecs['vc_area'] = tvc_area
    tspecs['mi_num'] = tmi_num
    tspecs['sj_num'] = tsj_num
    tspecs['vc_num'] = tvc_num

    tlabels = info['labels']
    tcell_mask = tlabels[0] < 7
    torga_mask = tlabels[0] > 6
    tcell_vertex_num = tlabels[1][tcell_mask.reshape(-1)].sum()
    torga_vertex_num = tlabels[1][torga_mask.reshape(-1)].sum()
    assert tcell_vertex_num + torga_vertex_num == info['vertex_num']

    if 1 in info['types'][0]:
        myelin = info['types'][1][1] / tcell_vertex_num
    else:
        myelin = 0
    info_txt += f'\nmyelin coverage: {round(myelin*100, 1)}'
    tspecs['my'] = myelin

    for ix, label in enumerate(tlabels[0]):
        if label < 7:
            cov = tlabels[1][ix] / tcell_vertex_num
            info_txt += f'\n{classes[label]} ({label}) percentage: {round(cov*100, 1)}'
            tspecs[label] = cov

    info_txt += f"\ntotal vertex number: {info['vertex_num']}"
    info_txt += f"\ncell vertex number: {tcell_vertex_num}"
    info_txt += f"\nnode number: {info['node_num']}"
    tspecs['vertex_num'] = info['vertex_num']
    tspecs['cell_vertex_num'] = tcell_vertex_num
    tspecs['node_num'] = info['node_num']

    info_txt += f"\ntotal & {round(tcell_area)} & {info['node_num']} & " \
                f"{round(myelin * 100, 1)} & {round(tspecs[0] * 100, 1)} & {round(tspecs[1] * 100, 1)} & " \
                f"{round(tspecs[2] * 100, 1)} & {round(tspecs[3] * 100, 1)} & {round(tspecs[4] * 100, 1)} & " \
                f"{round(tspecs[5] * 100, 1)} & {round(tspecs[6] * 100, 1)} \\"

    info_txt += f"\ntotal & {round(tmi_area)} ({tmi_num}) & {round(tsj_area)} ({tsj_num}) & " \
                f"{round(tvc_area)} ({tvc_num}) \\"

    with open(out_path + name + '.txt', 'a') as f:
        f.write(info_txt)
        f.close()

    with open(out_path + name + '.pkl', 'wb') as f:
        pickle.dump(tspecs, f)
        f.close()


def eval_gt(data_path: str, out_path: str, name: str):
    data_path = os.path.expanduser(data_path)
    out_path = os.path.expanduser(out_path)
    data = basics.load_pkl(data_path)
    area_type = []
    areas = []
    types = []
    percs = []
    ttypes = []
    tpercs = []
    for key in data:
        if isinstance(key, str) and 'sso' in key:
            curr = data[key]
            types.append(1)
            percs.append(curr['my']*100)
            for i in range(7):
                types.append(i+2)
                percs.append(curr[i]*100)
            areas.append(curr['cell_area'])
            area_type.append(1)
            areas.append(curr['mi_area'])
            area_type.append(2)
            areas.append(curr['sj_area'])
            area_type.append(3)
            areas.append(curr['vc_area'])
            area_type.append(4)
    ttypes.append(1)
    tpercs.append(data['my'] * 100)
    for i in range(7):
        ttypes.append(i + 2)
        tpercs.append(data[i] * 100)
    fontsize = 20
    mpl.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots()
    ax.scatter(types, percs, c='k', marker='o', s=10, zorder=3, label='SSVs')
    ax.scatter(ttypes, tpercs, c='b', marker='o', s=30, zorder=4, label='total')
    ax.grid(zorder=0)
    ax.set_ylabel('percentage %', fontsize=fontsize)
    ax.legend(loc=0, fontsize=fontsize)
    plt.ylim(top=100)
    plt.tight_layout()
    plt.yticks(fontsize=fontsize)
    plt.xticks(types, ['my', 'de', 'ax', 'so', 'bo', 'te', 'ne', 'he'], fontsize=fontsize)
    plt.savefig(out_path + name + '_types.eps')
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(area_type, areas, c='k', marker='o', s=10, zorder=3)
    ax.grid(zorder=0)
    ax.set_ylabel('area in \u03BCm²', fontsize=fontsize, labelpad=20)
    plt.yscale('symlog')
    plt.tight_layout()
    plt.ylim(bottom=0)
    plt.yticks(fontsize=fontsize)
    plt.xticks(area_type, ['cell', 'mi', 'sj', 'vc'], fontsize=fontsize)
    plt.savefig(out_path + name + '_areas.eps')
    plt.close()


def add_myelin(set_path: str, out_path: str):
    """ loads myelin predictions, maps them to node coordinates, then maps them to vertices and saves all
        CloudEnsembles at given path with the myelin predicitons included to out_path. """
    set_path = os.path.expanduser(set_path)
    out_path = os.path.expanduser(out_path)
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    files = glob.glob(set_path + '*.pkl')
    for file in tqdm(files):
        sso_id = int(re.findall(r"/sso_(\d+).", file)[0])
        sso = SuperSegmentationObject(sso_id)
        sso.load_skeleton()
        myelinated = map_myelin2coords(sso.skeleton["nodes"], mag=4)
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


if __name__ == '__main__':
    # produce_set_info()
    # eval_set_info('/u/jklimesch/thesis/results/gt/train_info.pkl', '/u/jklimesch/thesis/results/gt/', 'train_gt2')
    # eval_set_info('/u/jklimesch/thesis/results/gt/eval_info.pkl', '/u/jklimesch/thesis/results/gt/', 'eval_gt2')
    eval_gt('~/thesis/results/gt/eval_gt2.pkl', '~/thesis/results/gt/', 'eval')
    eval_gt('~/thesis/results/gt/train_gt2.pkl', '~/thesis/results/gt/', 'train')
