import torch
import time
from elektronn3.models.convpoint_test import SegBig, indices_conv_reduction
from elektronn3.models.convpoint import indices_conv_reduction as icr2

input_channels = 4
nclasses = 7
batch_size = 2
npoints = 10000


with torch.no_grad():
    device = torch.device('cuda')
    model = SegBig(input_channels, nclasses)
    model.to(device)

    pts = torch.rand(batch_size, npoints, 3)
    pts[0, -4:, :] = torch.ones((1, 3))*1000
    pts[1, -7:, :] = torch.ones((1, 3))*1000
    feats = torch.ones(batch_size, npoints, input_channels)
    pts = pts.to(device)
    feats = feats.to(device)

    # outputs = model(feats, pts)

    total_time = 0
    for i in range(100):
        start = time.time()
        indices_conv_reduction(pts, 2048, 16)
        total_time += time.time()-start
    print(total_time/20)

    total_time = 0
    for i in range(100):
        start = time.time()
        icr2(pts, 2048, 16)
        total_time += time.time()-start
    print(total_time/20)
