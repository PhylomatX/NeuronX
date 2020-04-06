import os
import torch
import numpy as np
from morphx.classes.pointcloud import PointCloud
from elektronn3.models.convpoint import SegBig
from neuronx.classes.argscontainer import ArgsContainer


class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output[0]

    def close(self):
        self.hook.remove()


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
    analyse_layer('~/thesis/results/ads_prediction/2020_03_25_40000_80000_co/', 12, 0, 300,
                  '~/thesis/results/ads_prediction/model_analysis/sample.pkl')
