import os
import re
import glob
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import basics, objects
from syconn.reps.super_segmentation import SuperSegmentationDataset


def preds2hc(preds: str):
    preds = os.path.expanduser(preds)
    preds = basics.load_pkl(preds)
    obj = objects.load_obj('ce', preds[0])
    obj.set_predictions(preds[1])
    obj.generate_pred_labels()
    if isinstance(obj, CloudEnsemble):
        hc = obj.hc
    else:
        hc = obj
    _ = hc.pred_node_labels
    return hc


def preds2kzip(pred_folder: str, out_path: str, ssd_path: str, col_lookup: dict,
               label_mappings: Optional[List[Tuple[int, int]]] = None):
    pred_folder = os.path.expanduser(pred_folder)
    out_path = os.path.expanduser(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    files = glob.glob(pred_folder + '*_preds.pkl')
    ssd = SuperSegmentationDataset(ssd_path)
    for file in tqdm(files):
        hc_voxeled = preds2hc(file)
        sso_id = int(re.findall(r"/sso_(\d+).", file)[0])
        sso = ssd.get_super_segmentation_object(sso_id)

        verts = sso.mesh[1].reshape(-1, 3)
        hc = HybridCloud(nodes=hc_voxeled.nodes, edges=hc_voxeled.edges, node_labels=hc_voxeled.node_labels,
                         pred_node_labels=hc_voxeled.pred_node_labels, vertices=verts)
        hc.nodel2vertl()
        hc.prednodel2predvertl()
        if label_mappings is not None:
            hc.map_labels(label_mappings)

        cols = np.array([col_lookup[el] for el in hc.pred_labels.squeeze()], dtype=np.uint8)
        sso.mesh2kzip(out_path + f'p_{sso_id}.k.zip', ext_color=cols)
        cols = np.array([col_lookup[el] for el in hc.labels.squeeze()], dtype=np.uint8)
        sso.mesh2kzip(out_path + f't_{sso_id}.k.zip', ext_color=cols)

        comments = list(hc.pred_node_labels.reshape(-1))
        for node in range(len(hc.nodes)):
            if hc.pred_node_labels[node] != hc.node_labels[node] and hc.pred_node_labels[node] != -1:
                comments[node] = 'e' + str(comments[node])
        sso.save_skeleton_to_kzip(out_path + f'p_{sso_id}.k.zip', comments=comments)
        comments = hc.node_labels.reshape(-1)
        sso.save_skeleton_to_kzip(out_path + f't_{sso_id}.k.zip', comments=comments)


colors = {0: (125, 125, 125, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255), 3: (125, 125, 255, 255),
          4: (255, 255, 125, 255), 5: (125, 255, 255, 255), 6: (255, 125, 255, 255), -1: (0, 0, 0, 255)}

preds2kzip('~/thesis/current_work/paper/7_class/2020_10_19_8000_8192_7class_cp_cp_fps/eval_40_valiter1_batchsize-1/'
           'epoch_40/',
           '~/thesis/current_work/paper/7_class/2020_10_19_8000_8192_7class_cp_cp_fps/eval_40_valiter1_batchsize-1/'
           'kzips/',
           "/wholebrain/songbird/j0126/areaxfs_v6/", colors)
