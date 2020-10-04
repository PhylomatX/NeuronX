import os
import open3d
import torch
import numpy as np
from morphx.classes.pointcloud import PointCloud
from morphx.processing import clouds
from elektronn3.models.convpoint import SegBig, SegAdapt
from neuronx.classes.argscontainer import ArgsContainer


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


def analyse_features(m_path: str, args_path: str, out_path: str):
    device = torch.device('cuda')
    m_path = os.path.expanduser(m_path)
    out_path = os.path.expanduser(out_path)
    args_path = os.path.expanduser(args_path)

    # load model specifications
    argscont = ArgsContainer().load_from_pkl(args_path)

    # load model
    if argscont.use_big:
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

    layer_outs = SaveFeatures(list(model.children())[1])
    act_outs = SaveFeatures([list(model.children())[0]])

    pts = torch.rand(1, argscont.sample_num, 3, device=device)
    feats = torch.rand(1, argscont.sample_num, argscont.input_channels, device=device)
    _ = model(feats, pts)

    for ix, layer in enumerate(layer_outs.features):
        if len(layer) < 2:
            continue
        feats = layer[0].detach().cpu().numpy()[0]
        feats_act = act_outs.features[ix].detach().cpu().numpy()[0]
        pts = layer[1].detach().cpu().numpy()[0]
        x_offset = (pts[:, 0].max() - pts[:, 0].min()) * 1.5 * 2
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
            pc_act.move(np.array([(i % 8) * x_offset + x_offset / 2.5, y_offset, 0]))
            pc = clouds.merge_clouds([pc, pc_act])
            if total_pc is None:
                total_pc = pc
                total_pc_act = pc_act
            else:
                total_pc = clouds.merge_clouds([total_pc, pc])
                total_pc_act = clouds.merge_clouds([total_pc_act, pc_act])
        total_pc.move(np.array([-4 * x_offset - x_offset / 2, -row_num / 2 * y_size - y_size / 2, 0]))
        total_pc_act.move(np.array([-4 * x_offset - x_offset / 2, -row_num / 2 * y_size - y_size / 2, 0]))
        total_pc.save2pkl(out_path + f'layer_{ix}.pkl')
        total_pc_act.save2pkl(out_path + f'layer_{ix}_act.pkl')


def analyse_layer(model_path: str, layer: int, filter: int, opt_steps: int, out_path: str):
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
    analyse_features('~/thesis/current_work/paper/ads_thesis/2020_09_22_12000_12000_small/models/state_dict_e200.pth',
                     '~/thesis/current_work/paper/ads_thesis/2020_09_22_12000_12000_small/argscont.pkl',
                     '~/thesis/current_work/paper/model_analysis/')
