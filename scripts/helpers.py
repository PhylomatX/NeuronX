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


def produce_chunks():
    """ Create and save all resulting chunks of an dataset. """
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    chunk_size = 10000
    # train_transforms = [clouds.RandomVariation((-100, 100)), clouds.RandomShear(limits=(-0.3, 0.3)),
    #                     clouds.RandomRotate(), clouds.Normalization(chunk_size), clouds.Center()]
    train_transforms = clouds.Compose([clouds.Identity()])
    path = os.path.expanduser('~/thesis/tmp/poisson/')
    save_path = f'{path}examples/'
    ch = ChunkHandler(path, sample_num=10000, density_mode=False, tech_density=100, bio_density=100, specific=True,
                      chunk_size=chunk_size, obj_feats=features, transform=train_transforms, splitting_redundancy=2,
                      label_mappings=[(5, 1), (6, 2)], label_remove=[1, 2, 3, 4], sampling=False)
    vert_nums = []
    for item in ch.obj_names:
        total = None
        for i in range(ch.get_obj_length(item)):
            sample, idcs, vert_num = ch[(item, i)]
            vert_nums.append(vert_num)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                os.makedirs(f'{save_path}vertnums/')
            with open(f'{save_path}{item}_{i}.pkl', 'wb') as f:
                pickle.dump([sample, sample], f)
            f.close()
            if total is None:
                total = sample
            else:
                total = clouds.merge_clouds([total, sample])
        with open(f'{save_path}{item}_total.pkl', 'wb') as f:
            pickle.dump(total, f)
        f.close()
    vert_nums = np.array(vert_nums)
    print(f"Min: {vert_nums.min()}")
    print(f"Max: {vert_nums.max()}")
    print(f"Mean: {vert_nums.mean()}")
    with open(f'{save_path}vertnums/{chunk_size}_vertnums.pkl', 'wb') as f:
        pickle.dump(vert_nums, f)
    f.close()


def compare_chunks():
    """ Create chunks with different ChunkHandlers and compare the results. """
    path = os.path.expanduser('~/thesis/gt/test_gt/')
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    transforms1 = clouds.Compose([clouds.Center(), clouds.RandomShear(limits=(-0.5, 0.5))])
    ch1 = ChunkHandler(path, sample_num=10000, density_mode=False, specific=True, chunk_size=10000,
                       obj_feats=features, transform=transforms1)
    transforms2 = clouds.Compose([clouds.Center()])
    ch2 = ChunkHandler(path, sample_num=10000, density_mode=False, specific=True, chunk_size=10000,
                       obj_feats=features, transform=transforms2)
    item = ch1.obj_names[2]
    for i in range(4):
        sample1, _ = ch1[(item, i)]
        sample2, _ = ch2[(item, i)]
        samples = [sample1, sample2]
        with open(f'{path}shear_vis/{item}_{i}_05.pkl', 'wb') as f:
            pickle.dump(samples, f)
        f.close()


if __name__ == '__main__':
    produce_chunks()
