import os
import torch
import random
import pickle
import ipdb
import warnings
import numpy as np
from torch import nn
from datetime import date
from syconn.mp import batchjob_utils as qu
import morphx.processing.clouds as clouds
from morphx.data.torchhandler import TorchHandler
from morphx.data.chunkhandler import ChunkHandler
from morphx.postprocessing.mapping import PredictionMapper
from syconn import global_params
# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
from elektronn3.models.convpoint import SegSmall, SegBig
from elektronn3.training import Trainer3d, Backup


def start_trainings():
    today = date.today().strftime("%Y_%m_%d")
    multi_params = []

    for radius in [1000, 8000, 15000, 25000]:
        for npoints in [1000, 5000, 10000]:
            args = ['/u/jklimesch/thesis/trainings/current/',                       # save_root
                    '/u/jklimesch/thesis/gt/gt_poisson/ads/',                       # train_path
                    radius,                                                         # radius
                    npoints,                                                        # npoints
                    today + '_{}'.format(radius) + '_{}'.format(npoints),           # name
                    3,                                                              # nclasses
                    [clouds.RandomVariation((-10, 10)),
                     clouds.RandomRotate(),
                     clouds.Center()],
                    [clouds.Center()],                                              # val transforms
                    16,                                                             # batch_size
                    True,                                                           # use_cuda
                    1,                                                              # input_channels
                    True,                                                           # use_big
                    0,                                                              # random_seed
                    '/u/jklimesch/thesis/gt/gt_poisson/ads/single/',                # val_path
                    False,                                                          # track_running_stats
                    False                                                           # validation
                    ]
            multi_params.append(args)

    _ = qu.QSUB_script(multi_params, "mx_run_e3_training", n_cores=10, remove_jobfolder=True,
                       additional_flags="--gres=gpu:1 --time=7-0 --mem=125000")


def training_thread(args):
    # read parameters from args
    print(args)
    save_root = os.path.expanduser(args[0])
    train_path = os.path.expanduser(args[1])
    radius = args[2]
    npoints = args[3]
    name = args[4]
    nclasses = args[5]
    train_transforms = args[6]
    val_transforms = args[7]
    batch_size = args[8]
    use_cuda = args[9]
    input_channels = args[10]
    use_big = args[11]
    random_seed = args[12]
    val_path = args[13]
    trs = args[14]
    validation = args[15]

    # define other parameters
    lr = 1e-3
    lr_stepsize = 1000
    lr_dec = 0.995
    max_steps = 500000
    jit = False

    # set random seeds to ensure compareability of different trainings
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    args_cache = args.copy()

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # create network and dataset
    if use_big:
        model = SegBig(input_channels, nclasses, trs=trs)
    else:
        model = SegSmall(input_channels, nclasses)

    if use_cuda:
        if torch.cuda.device_count() > 1:
            batch_size = batch_size * torch.cuda.device_count()
            model = nn.DataParallel(model)
        model.to(device)

    # Example for a model-compatible input.
    example_pts = torch.ones(16, 1000, 3).to(device)
    example_feats = torch.ones(16, 1000, 1).to(device)
    if jit:
        # Make sure that tracing works
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tracedmodel = torch.jit.trace(model, (example_feats, example_pts))

    train_transform = clouds.Compose(train_transforms)
    train_ds = TorchHandler(train_path, radius, npoints, train_transform)

    if validation:
        val_transform = clouds.Compose(val_transforms)
        val_ds = ChunkHandler(val_path, radius, npoints, val_transform, specific=True)
        pm = PredictionMapper(val_path, save_root + 'validation/' + name + '/', radius)
    else:
        val_ds = None
        pm = None

    # initialize optimizer, scheduler, loss and trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

    criterion = torch.nn.CrossEntropyLoss()
    if use_cuda:
        criterion.cuda()

    trainer = Trainer3d(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_ds,
        valid_dataset=val_ds,
        val_freq=10,
        pred_mapper=pm,
        batchsize=batch_size,
        num_workers=0,
        save_root=save_root,
        exp_name=name,
        num_classes=nclasses,
        schedulers={"lr": scheduler},
        example_input=(example_feats, example_pts),
        enable_save_trace=jit
    )

    # Archiving training script, src folder, env info
    Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()
    with open(trainer.save_path + '/training_args.pkl', 'wb') as f:
        pickle.dump(args_cache, f)

    # Start training
    trainer.run(max_steps)


if __name__ == '__main__':

    multi = False

    if multi:
        global_params.wd = "/u/jklimesch/thesis/trainings/mp/"
        start_trainings()
    else:
        today = date.today().strftime("%Y_%m_%d")
        chunk_size = 15000
        sample_num = 20000
        args = ['/u/jklimesch/thesis/trainings/current/',  # save_root
                '/u/jklimesch/thesis/gt/gt_ensembles/ads/single/',  # train_path
                chunk_size,  # radius
                sample_num,  # npoints
                today + '_{}'.format(chunk_size) + '_{}'.format(sample_num),  # name
                3,  # nclasses
                [clouds.RandomVariation((-50, 50)),
                 clouds.RandomRotate(),
                 clouds.Normalization(chunk_size),
                 clouds.Center()],
                [clouds.Center()],  # val transforms
                4,  # batch_size
                True,  # use_cuda
                1,  # input_channels
                True,  # use_big
                0,  # random_seed
                '/u/jklimesch/thesis/gt/gt_ensembles/ads/',  # val_path
                False,  # track_running_stats
                False   # validation
                ]
        training_thread(args)
