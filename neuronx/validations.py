import os
import torch
import random
import numpy as np
from morphx.processing import clouds
from morphx.data.torchset import TorchSet


def validation_thread(args):
    save_root = os.path.expanduser(args[0])
    data_path = os.path.expanduser(args[1])
    radius = args[2]
    npoints = args[3]
    name = args[4]
    nclasses = args[5]
    transforms = args[6]
    batch_size = args[7]
    use_cuda = args[8]
    input_channels = args[9]
    use_big = args[10]
    random_seed = args[11]

    # set random seeds to ensure compareability of different trainings
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load scripted model
    model_path = save_root + '/' + name + '/model.pts'
    model = torch.jit.load(model_path, map_location=device)

    transform = clouds.Compose(transforms)
    ds = TorchSet(data_path, radius, npoints, transform, class_num=nclasses)





