import torch
import time
from tqdm import tqdm
from elektronn3.models.convpoint import SegTest

input_channels = 1
nclasses = 3
device = torch.device('cuda')

model = SegTest(input_channels, nclasses)
model.to(device)

pts = torch.rand(16, 10000, 3)
feats = torch.ones(16, 10000, 1)

pts = pts.to(device)
feats = feats.to(device)

model_time = 0
runs = 5

for i in tqdm(range(runs)):
    start = time.time()
    outputs = model(feats, pts)
    model_time += time.time() - start

print(model_time/runs)
