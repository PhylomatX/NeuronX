import os
import time
import torch
import pickle
import gpustat
from tqdm import tqdm
from threading import Thread
from elektronn3.models.convpoint import SegBig

input_channels = 4
nclasses = 7
batch_size = 256
npoints = 10

with torch.no_grad():
    device = torch.device('cuda')
    model = SegBig(input_channels, nclasses)
    model.to(device)

    pts = torch.rand(batch_size, npoints, 3)
    feats = torch.ones(batch_size, npoints, input_channels)
    pts = pts.to(device)
    feats = feats.to(device)

    model_time = 0
    runs = 3

    for i in tqdm(range(runs)):
        start = time.time()
        outputs = model(feats, pts)
        model_time += time.time() - start
        print(gpustat.new_query())

    print(model_time/runs)

