import os
import open3d
import torch
import numpy as np
from typing import List, Tuple
from morphx.classes.pointcloud import PointCloud
from morphx.processing import clouds, basics
from elektronn3.models.convpoint import SegBig, SegAdapt
from neuronx.classes.argscontainer import ArgsContainer
from neuronx.classes.torchhandler import TorchHandler
from lightconvpoint.utils import get_network


class SaveFeatures:
    def __init__(self, modules):
        self.hooks = []
        for module in modules:
            self.hooks.append(module.register_forward_hook(self.hook_fn))
        self.features = []

    def hook_fn(self, module, input, output):
        self.features.append(output)

    def close(self):
        for hook in self.hooks:
            hook.remove()


def analyse_features(m_path: str, args_path: str, out_path: str, val_path: str, context_list: List[Tuple[str, int]],
                     label_mappings: List[Tuple[int, int]] = None, label_remove: List[int] = None,
                     splitting_redundancy: int = 1, test: bool = False):
    device = torch.device('cuda')
    m_path = os.path.expanduser(m_path)
    out_path = os.path.expanduser(out_path)
    args_path = os.path.expanduser(args_path)
    val_path = os.path.expanduser(val_path)

    # load model specifications
    argscont = ArgsContainer().load_from_pkl(args_path)

    lcp_flag = False
    # load model
    if argscont.architecture == 'lcp' or argscont.model == 'ConvAdaptSeg':
        kwargs = {}
        if argscont.model == 'ConvAdaptSeg':
            kwargs = dict(f_map_num=argscont.pl, architecture=argscont.architecture, act=argscont.act, norm=argscont.norm_type)
        conv = dict(layer=argscont.conv[0], kernel_separation=argscont.conv[1])
        model = get_network(argscont.model, argscont.input_channels, argscont.class_num, conv, argscont.search, **kwargs)
        lcp_flag = True
    elif argscont.use_big:
        model = SegBig(argscont.input_channels, argscont.class_num, trs=argscont.track_running_stats, dropout=0,
                       use_bias=argscont.use_bias, norm_type=argscont.norm_type, use_norm=argscont.use_norm,
                       kernel_size=argscont.kernel_size, neighbor_nums=argscont.neighbor_nums,
                       reductions=argscont.reductions, first_layer=argscont.first_layer,
                       padding=argscont.padding, nn_center=argscont.nn_center, centroids=argscont.centroids,
                       pl=argscont.pl, normalize=argscont.cp_norm)
    else:
        print("Adaptable model was found!")
        model = SegAdapt(argscont.input_channels, argscont.class_num, architecture=argscont.architecture,
                         trs=argscont.track_running_stats, dropout=argscont.dropout, use_bias=argscont.use_bias,
                         norm_type=argscont.norm_type, kernel_size=argscont.kernel_size, padding=argscont.padding,
                         nn_center=argscont.nn_center, centroids=argscont.centroids, kernel_num=argscont.pl,
                         normalize=argscont.cp_norm, act=argscont.act)
    try:
        full = torch.load(m_path)
        model.load_state_dict(full)
    except RuntimeError:
        model.load_state_dict(full['model_state_dict'])
    model.to(device)
    model.eval()

    pts = torch.rand(1, argscont.sample_num, 3, device=device)
    feats = torch.rand(1, argscont.sample_num, argscont.input_channels, device=device)
    contexts = []
    th = None

    if not test:
        # prepare data loader
        if label_mappings is None:
            label_mappings = argscont.label_mappings
        if label_remove is None:
            label_remove = argscont.label_remove
        transforms = clouds.Compose(argscont.val_transforms)
        th = TorchHandler(val_path, argscont.sample_num, argscont.class_num, density_mode=argscont.density_mode,
                          bio_density=argscont.bio_density, tech_density=argscont.tech_density, transform=transforms,
                          specific=True, obj_feats=argscont.features, ctx_size=argscont.chunk_size,
                          label_mappings=label_mappings, hybrid_mode=argscont.hybrid_mode,
                          feat_dim=argscont.input_channels, splitting_redundancy=splitting_redundancy,
                          label_remove=label_remove, sampling=argscont.sampling,
                          force_split=False, padding=argscont.padding, exclude_borders=0)
        for context in context_list:
            pts = torch.zeros((1, argscont.sample_num, 3))
            feats = torch.ones((1, argscont.sample_num, argscont.input_channels))
            sample = th[context]
            pts[0] = sample['pts']
            feats[0] = sample['features']
            o_mask = sample['o_mask'].numpy().astype(bool)
            l_mask = sample['l_mask'].numpy().astype(bool)
            target = sample['target'].numpy()
            target = target[l_mask].astype(int)
            contexts.append((feats, pts, o_mask, l_mask, target))
    else:
        contexts.append((feats, pts))

    for c_ix, context in enumerate(contexts):
        # set hooks

        if lcp_flag:
            layer_outs = SaveFeatures(list(model.children())[0][1:])
            act_outs = SaveFeatures([layer.activation for layer in list(model.children())[0][1:]])
        else:
            layer_outs = SaveFeatures(list(model.children())[1])
            act_outs = SaveFeatures([list(model.children())[0]])
        feats = context[0].to(device, non_blocking=True)
        pts = context[1].to(device, non_blocking=True)

        if lcp_flag:
            pts = pts.transpose(1, 2)
            feats = feats.transpose(1, 2)

        output = model(feats, pts).cpu().detach()

        if lcp_flag:
            output = output.transpose(1, 2).numpy()

        if not test:
            output = np.argmax(output[0][context[2]].reshape(-1, th.num_classes), axis=1)
            pts = context[1][0].numpy()
            identifier = f'{context_list[c_ix][0]}_{context_list[c_ix][1]}'
            target = PointCloud(pts, context[4])
            x_offset = (pts[:, 0].max() - pts[:, 0].min()) * 1.5 * 3
            pred = PointCloud(pts[context[3]], output)
            pred.move(np.array([x_offset / 2, 0, 0]))
            clouds.merge([target, pred]).save2pkl(out_path + identifier + '_0io_r_a.pkl')
        for ix, layer in enumerate(layer_outs.features):
            if len(layer) < 2:
                continue
            feats = layer[0].detach().cpu()[0]
            feats_act = act_outs.features[ix].detach().cpu()[0]
            pts = layer[1].detach().cpu()[0]
            if lcp_flag:
                feats = feats.transpose(0, 1).numpy()
                feats_act = feats_act.transpose(0, 1).numpy()
                pts = pts.transpose(0, 1).numpy()
            else:
                feats = feats.numpy()
                feats_act = feats_act.numpy()
                pts = pts.numpy()
            x_offset = (pts[:, 0].max() - pts[:, 0].min()) * 1.5 * 3
            x_offset_act = x_offset / 3
            y_size = (pts[:, 1].max() - pts[:, 1].min()) * 1.5
            y_offset = 0
            row_num = feats.shape[1] / 8
            total_pc = None
            total_pc_act = None
            for i in range(feats.shape[1]):
                if i % 8 == 0 and i != 0:
                    y_offset += y_size
                pc = PointCloud(vertices=pts, features=feats[:, i].reshape(-1, 1))
                pc_act = PointCloud(vertices=pts, features=feats_act[:, i].reshape(-1, 1))
                pc.move(np.array([(i % 8) * x_offset, y_offset, 0]))
                pc_act.move(np.array([(i % 8) * x_offset + x_offset / 2.8, y_offset, 0]))
                pc = clouds.merge_clouds([pc, pc_act])
                pc_act = PointCloud(vertices=pts, features=feats_act[:, i].reshape(-1, 1))
                pc_act.move(np.array([(i % 8) * x_offset_act, y_offset, 0]))
                if total_pc is None:
                    total_pc = pc
                    total_pc_act = pc_act
                else:
                    total_pc = clouds.merge_clouds([total_pc, pc])
                    total_pc_act = clouds.merge_clouds([total_pc_act, pc_act])
            total_pc.move(np.array([-4 * x_offset - x_offset / 2, -row_num / 2 * y_size - y_size / 2, 0]))
            total_pc_act.move(np.array([-4 * x_offset_act - x_offset_act / 2, -row_num / 2 * y_size - y_size / 2, 0]))
            total_pc.save2pkl(out_path + f'{context_list[c_ix][0]}_{context_list[c_ix][1]}_l{ix}_r.pkl')
            total_pc_act.save2pkl(out_path + f'{context_list[c_ix][0]}_{context_list[c_ix][1]}_l{ix}_a.pkl')


def fit_cloud2layer(model_path: str, layer: int, filter: int, opt_steps: int, out_path: str):
    device = torch.device('cuda')
    model_path = os.path.expanduser(model_path)
    out_path = os.path.expanduser(out_path)
    # load model specifications
    argscont = ArgsContainer().load_from_pkl(model_path + '/argscont.pkl')

    # load model
    model = SegBig(argscont.input_channels, argscont.class_num)
    full = torch.load(model_path + 'state_dict.pth')
    model.load_state_dict(full['model_state_dict'])
    model.to(device)
    model.eval()

    activations = SaveFeatures(list(model.children())[layer])

    pts = torch.rand(1, argscont.sample_num, 3, device=device, requires_grad=True)
    feats = torch.rand(1, argscont.sample_num, argscont.input_channels, device=device)
    feats[0][:] = torch.tensor([1, 0, 0, 0])
    num_orga = int(0.2*len(feats[0]))
    feats[0][:num_orga] = torch.tensor([0, 1, 0, 0])
    feats = feats.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([pts], lr=0.1, weight_decay=1e-6)

    for n in range(opt_steps):
        optimizer.zero_grad()
        _ = model(feats, pts)
        loss = -activations.features[0, :, filter].mean()
        loss.backward()
        optimizer.step()
        print(loss)

    pts = pts.cpu().detach().numpy()[0]
    labels = np.ones((len(feats[0]), 1))
    labels[:] = 10
    labels[:num_orga] = 7

    pc = PointCloud(pts, labels)
    pc.save2pkl(out_path)


if __name__ == '__main__':
    analyse_features(m_path='~/thesis/current_work/paper/dnh/2020_10_15_8000_8192_cp_cp_r/models/state_dict_e570.pth',
                     args_path='~/thesis/current_work/paper/dnh/2020_10_15_8000_8192_cp_cp_r/argscont.pkl',
                     out_path='~/thesis/current_work/paper/model_analysis/2020_10_15_8000_8192_cp_cp_r/',
                     val_path='~/thesis/gt/cmn/dnh/voxeled/evaluation/',
                     context_list=[('sso_8003584', 1), ('sso_8003584', 2), ('sso_8003584', 3)],
                     label_remove=[1, 2, 3, 4],  label_mappings=[(5, 1), (6, 2)])
