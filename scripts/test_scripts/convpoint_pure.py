import os
import time
import torch
import pickle
import gpustat
from tqdm import tqdm
from threading import Thread
from elektronn3.models.convpoint import SegBig


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls
        self.queries = []
        self.start()

    def run(self):
        while not self.stopped:
            self.queries.append(gpustat.new_query())
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


input_channels = 4
nclasses = 7
batch_size = 8
npoints = 28000

# Instantiate monitor with a 10-second delay between updates
monitor = Monitor(0.1)

device = torch.device('cuda')
model = SegBig(input_channels, nclasses)
model.to(device)

pts = torch.rand(batch_size, npoints, 3)
feats = torch.ones(batch_size, npoints, input_channels)
pts = pts.to(device)
feats = feats.to(device)

model_time = 0
runs = 30

for i in tqdm(range(runs)):
    start = time.time()
    outputs = model(feats, pts)
    model_time += time.time() - start

print(model_time/runs)

monitor.stop()
print(monitor.queries[-1])

f = open(f'/u/jklimesch/thesis/convpoint_capacity/ch{input_channels}_cl{nclasses}_bs{batch_size}_np{npoints}.pkl', 'wb')
pickle.dump(monitor.queries, f)
f.close()
