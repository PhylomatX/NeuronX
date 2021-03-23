import os
import pickle as pkl
import numpy as np
import sklearn.metrics as sm
from typing import Dict, Tuple, List
from neuronx.pipeline.evaluate import get_target_names, remove_points_without_prediction, write_confusion_matrix
from morphx.processing import objects, basics
from morphx.classes.pointcloud import PointCloud

red = 5


def merge(base: np.ndarray, replace: Dict[int, Tuple[np.ndarray, List[Tuple[int, int]]]]):
    for key in replace:
        replace_labels = replace[key][0]
        for mapping in replace[key][1]:
            replace_labels[replace_labels == mapping[0]] = mapping[1]
        mask = (base == key).reshape(-1)
        base[mask] = replace_labels[mask]
    return base


def get_pred_verts(verts: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (preds != -1).reshape(-1)
    return verts[mask], preds[mask]


if __name__ == '__main__':
    base_path = os.path.expanduser(f'~/working_dir/paper/hierarchy/')

    sso_ids = [491527, 12179464, 14141444, 18251791, 22335491, 23044610, 46319619]
    # sso_ids = [491527, 12179464]

    total_labels = {'pred': np.array([]), 'pred_node': np.array([]), 'gt': np.array([]), 'gt_node': np.array([]),
                    'coverage': [0, 0]}
    reports_txt = "Confusion matrix: row = true label, column = predicted label \n" \
                  "Precision: What percentage of points from one label truly belong to that label \n" \
                  "Recall: What percentage of points from one true label have been predicted as that label \n\n"
    reports = {}
    target_names = ['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head']

    for sso_id in sso_ids:
        reports_txt += str(sso_id) + '\n\n'

        # 0: axon, 1: bouton, 2: terminal
        with open(base_path + f'abt/20_09_27_test_eval_red{red}_valiter1_batchsize-1/epoch_370/sso_' + str(sso_id) + '_preds.pkl', 'rb') as f:
            abt_preds = pkl.load(f)
            abt = objects.load_obj('ce', abt_preds[0])
            abt.set_predictions(abt_preds[1])
            abt.generate_pred_labels()
        # 0: dendrite, 1: neck, 2: head
        with open(base_path + f'dnh/20_09_27_test_eval_red{red}_valiter1_batchsize-1/epoch_390/sso_' + str(sso_id) + '_preds.pkl', 'rb') as f:
            dnh_preds = pkl.load(f)
            dnh = objects.load_obj('ce', dnh_preds[0])
            dnh.set_predictions(dnh_preds[1])
            dnh.generate_pred_labels()
        # 0: dendrite, 1: axon, 2: soma,
        with open(base_path + f'ads/20_09_27_test_eval_red{red}_nokdt_valiter1_batchsize-1/epoch_760/sso_' + str(sso_id) + '_preds.pkl', 'rb') as f:
            ads_preds = pkl.load(f)
            ads = objects.load_obj('ce', ads_preds[0])
            ads.set_predictions(ads_preds[1])
            ads.generate_pred_labels()

        # ads.hc.save2pkl(base_path + f'/merged/ads_{sso_id}.pkl')
        # abt.hc.save2pkl(base_path + f'/merged/abt_{sso_id}.pkl')
        # dnh.hc.save2pkl(base_path + f'/merged/dnh_{sso_id}.pkl')

        preds = merge(ads.hc.pred_labels, {1: (abt.hc.pred_labels, [(1, 3), (2, 4), (0, 1)]), 0: (dnh.hc.pred_labels, [(1, 5), (2, 6)])})
        obj = objects.load_obj('ce', abt_preds[0])
        obj.remove_nodes([-2])
        hc = obj.hc
        hc.set_pred_labels(preds)

        verts, labels = get_pred_verts(hc.vertices, preds)
        pc = PointCloud(vertices=verts, labels=labels)
        pc.save2pkl(base_path + f'/merged/merged_{sso_id}.pkl')

        gtl, hcl = remove_points_without_prediction(hc.labels, hc.pred_labels, True)
        gtnl, hcnl = remove_points_without_prediction(hc.node_labels, hc.pred_node_labels, True)

        sso_report = {}
        mode = 'mv'

        coverage = hc.get_coverage()
        sso_report['cov'] = coverage

        # vertex-level eval
        targets = get_target_names(gtl, hcl, target_names)
        sso_report[mode] = sm.classification_report(gtl, hcl, output_dict=True, target_names=targets)
        reports_txt += mode + '\n\n' + f'Coverage: {coverage[1] - coverage[0]} of {coverage[1]}, ' \
                                       f'{round((1 - coverage[0] / coverage[1]) * 100)} %\n\n' + \
                       f'Number of predictions: {obj.pred_num}\n\n' + \
                       sm.classification_report(gtl, hcl, target_names=targets) + '\n\n'
        cm = sm.confusion_matrix(gtl, hcl)
        reports_txt += write_confusion_matrix(cm, targets) + '\n\n'

        # node-level eval
        mode += '_skel'
        targets = get_target_names(gtnl, hcnl, target_names)
        sso_report[mode] = sm.classification_report(gtnl, hcnl, output_dict=True, target_names=targets)
        reports_txt += mode + '\n\n' + sm.classification_report(gtnl, hcnl, target_names=targets) + '\n\n'
        cm = sm.confusion_matrix(gtnl, hcnl)
        reports_txt += write_confusion_matrix(cm, targets) + '\n\n\n\n\n'

        # save generated labels for total evaluation
        if total_labels is not None:
            total_labels['pred'] = np.append(total_labels['pred'], hcl)
            total_labels['pred_node'] = np.append(total_labels['pred_node'], hcnl)
            total_labels['gt'] = np.append(total_labels['gt'], gtl)
            total_labels['gt_node'] = np.append(total_labels['gt_node'], gtnl)
            total_labels['coverage'][0] += coverage[0]
            total_labels['coverage'][1] += coverage[1]

        reports[str(sso_id)] = sso_report

    total_report = {}
    total_report_txt = 'Total\n\n'
    mode = 'mv'

    coverage = total_labels['coverage']
    total_report['cov'] = coverage
    targets = get_target_names(total_labels['gt'], total_labels['pred'], target_names)
    total_report[mode] = \
        sm.classification_report(total_labels['gt'], total_labels['pred'], output_dict=True, target_names=targets)
    total_report_txt += \
        mode + '\n\n' + \
        f'Coverage: {coverage[1] - coverage[0]} of {coverage[1]}, ' \
        f'{round((1 - coverage[0] / coverage[1]) * 100)} %\n\n' + \
        sm.classification_report(total_labels['gt'], total_labels['pred'], target_names=targets) + '\n\n'
    cm = sm.confusion_matrix(total_labels['gt'], total_labels['pred'])
    total_report_txt += write_confusion_matrix(cm, targets) + '\n\n'
    mode += '_skel'
    targets = get_target_names(total_labels['gt_node'], total_labels['pred_node'], target_names)
    total_report[mode] = \
        sm.classification_report(total_labels['gt_node'], total_labels['pred_node'],
                                 output_dict=True, target_names=targets)
    total_report_txt += \
        mode + '\n\n' + \
        sm.classification_report(total_labels['gt_node'], total_labels['pred_node'], target_names=targets) + '\n\n'
    cm_skel = sm.confusion_matrix(total_labels['gt_node'], total_labels['pred_node'])
    total_report_txt += write_confusion_matrix(cm_skel, targets) + '\n\n'

    reports['total'] = total_report
    reports_txt += total_report_txt
    basics.save2pkl(reports, base_path, name='report')
    with open(base_path + 'report' + '.txt', 'w') as f:
        f.write(reports_txt)
