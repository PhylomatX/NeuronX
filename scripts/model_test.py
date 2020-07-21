import torch
import time
from elektronn3.models.convpoint_test import SegBig

input_channels = 4
nclasses = 7
batch_size = 8
npoints = 50

with torch.no_grad():
    device = torch.device('cuda')
    model = SegBig(input_channels, nclasses)
    model.to(device)

    for i in range(2):
        pts = torch.rand(batch_size, npoints, 3)
        feats = torch.ones(batch_size, npoints, input_channels)
        pts = pts.to(device)
        feats = feats.to(device)
        start = time.time()
        outputs = model(feats, pts)
