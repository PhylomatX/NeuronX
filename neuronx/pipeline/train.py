import torch
import random
import warnings
import numpy as np
from torch import nn
from datetime import date
import morphx.processing.clouds as clouds
from morphx.data.torchhandler import TorchHandler
from morphx.postprocessing.mapping import PredictionMapper
from syconn import global_params
# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
from elektronn3.models.convpoint import SegSmall, SegBig
from elektronn3.training import Trainer3d, Backup
from neuronx.classes.argscontainer import ArgsContainer
from elektronn3.training.schedulers import CosineAnnealingWarmRestarts


def training_thread(acont: ArgsContainer):
    # define other parameters
    lr = 1e-3
    lr_stepsize = 1000
    lr_dec = 0.995
    max_steps = int(acont.max_step_size / acont.batch_size)
    jit = False

    # set random seeds to ensure compareability of different trainings
    torch.manual_seed(acont.random_seed)
    np.random.seed(acont.random_seed)
    random.seed(acont.random_seed)

    if acont.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load model
    if acont.use_big:
        model = SegBig(acont.input_channels, acont.class_num, trs=acont.track_running_stats, dropout=0)
    else:
        model = SegSmall(acont.input_channels, acont.class_num)

    batch_size = acont.batch_size
    if acont.use_cuda:
        if torch.cuda.device_count() > 1:
            batch_size = acont.batch_size * torch.cuda.device_count()
            model = nn.DataParallel(model)
        model.to(device)

    # set up jit tracing
    example_pts = torch.ones(16, 1000, 3).to(device)
    example_feats = torch.ones(16, 1000, 1).to(device)
    if jit:
        # Make sure that tracing works
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tracedmodel = torch.jit.trace(model, (example_feats, example_pts))

    # set up environment
    train_transforms = clouds.Compose(acont.train_transforms)
    train_ds = TorchHandler(acont.train_path, acont.sample_num, acont.class_num, acont.input_channels,
                            density_mode=acont.density_mode,
                            chunk_size=acont.chunk_size,
                            bio_density=acont.bio_density,
                            tech_density=acont.tech_density,
                            transform=train_transforms,
                            obj_feats=acont.features,
                            label_mappings=acont.label_mappings,
                            hybrid_mode=acont.hybrid_mode)
    if acont.use_val:
        val_transforms = clouds.Compose(acont.val_transforms)
        val_ds = TorchHandler(acont.val_path, acont.sample_num, acont.class_num, acont.input_channels,
                              density_mode=acont.density_mode,
                              chunk_size=acont.chunk_size,
                              bio_density=acont.bio_density,
                              tech_density=acont.tech_density,
                              transform=val_transforms,
                              obj_feats=acont.features,
                              label_mappings=acont.label_mappings,
                              hybrid_mode=acont.hybrid_mode)

        # trainer3D was updated to new validation, so prediction mapping is not necessary any more.
        pm = None
        # pm = PredictionMapper(acont.val_path, f'{acont.save_root}validation/{name}/', splitfile=val_ds.splitfile)
    else:
        val_ds = None
        pm = None

    # initialize optimizer, scheduler, loss and trainer
    optimizer = None
    if acont.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif acont.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.5e-5)

    if optimizer is None:
        raise ValueError(f"Optimizer {acont.optimizer} is not known.")

    scheduler = None
    if acont.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    elif acont.scheduler == 'cosannwarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)

    if scheduler is None:
        raise ValueError(f"Scheduler {acont.scheduler} is not known.")

    criterion = torch.nn.CrossEntropyLoss()
    if acont.use_cuda:
        criterion.cuda()
    trainer = Trainer3d(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_ds,
        valid_dataset=val_ds,
        val_freq=acont.val_freq,
        val_iter=acont.val_iter,
        channel_num=acont.input_channels,
        pred_mapper=pm,
        batchsize=batch_size,
        num_workers=min(max(0, acont.batch_size-2), 5),
        save_root=acont.save_root,
        exp_name=acont.name,
        num_classes=acont.class_num,
        schedulers={"lr": scheduler},
        example_input=(example_feats, example_pts),
        enable_save_trace=jit
    )

    # Archiving training script, src folder, env info
    Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()
    acont.save2pkl(trainer.save_path + '/argscont.pkl')
    with open(trainer.save_path + '/argscont.txt', 'w') as f:
        f.write(str(argscont.attr_dict))
    f.close()

    # Start training
    trainer.run(max_steps)


if __name__ == '__main__':
    # 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6
    today = date.today().strftime("%Y_%m_%d")
    density_mode = False
    bio_density = 50
    sample_num = 28000
    chunk_size = 20000
    if density_mode:
        name = today + '_{}'.format(bio_density) + '_{}'.format(sample_num)
    else:
        name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)
    normalization = chunk_size
    argscont = ArgsContainer(save_root='/u/jklimesch/thesis/results/param_search_context/full/',
                             train_path='/u/jklimesch/thesis/gt/20_02_20/poisson_val_my/',
                             val_path='/u/jklimesch/thesis/gt/20_02_20/poisson_val/validation/',
                             sample_num=sample_num,
                             name=name + f'_full_my',
                             class_num=7,
                             train_transforms=[clouds.RandomVariation((-10, 10)), clouds.RandomRotate(),
                                               clouds.Normalization(normalization), clouds.Center()],
                             batch_size=8,
                             input_channels=5,
                             use_val=False,
                             val_freq=10,
                             features={'hc': {0: np.array([1, 0, 0, 0, 0]), 1: np.array([0, 1, 0, 0, 0])},
                                       'mi': np.array([0, 0, 1, 0, 0]),
                                       'vc': np.array([0, 0, 0, 1, 0]),
                                       'sy': np.array([0, 0, 0, 0, 1])},
                             chunk_size=chunk_size,
                             tech_density=1500,
                             bio_density=bio_density,
                             density_mode=density_mode,
                             max_step_size=10000000,
                             hybrid_mode=False,
                             scheduler='steplr',
                             optimizer='adam')
    training_thread(argscont)

    # # 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6
    # today = date.today().strftime("%Y_%m_%d")
    # density_mode = True
    # bio_density = 80
    # sample_num = 60000
    # chunk_size = 30000
    # if density_mode:
    #     name = today + '_{}'.format(bio_density) + '_{}'.format(sample_num)
    # else:
    #     name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)
    # normalization = 50000
    # argscont = ArgsContainer(save_root='/u/jklimesch/thesis/results/param_search_density/density_50/',
    #                          train_path='/u/jklimesch/thesis/gt/20_02_20/poisson_val_my/',
    #                          val_path='/u/jklimesch/thesis/gt/20_02_20/poisson_val/validation/',
    #                          sample_num=sample_num,
    #                          name=name + f'_co',
    #                          class_num=3,
    #                          train_transforms=[clouds.RandomVariation((-10, 10)), clouds.RandomRotate(),
    #                                            clouds.Normalization(normalization), clouds.Center()],
    #                          batch_size=2,
    #                          input_channels=4,
    #                          use_val=False,
    #                          val_freq=10,
    #                          features={'hc': np.array([1, 0, 0, 0]),
    #                                    'mi': np.array([0, 1, 0, 0]),
    #                                    'vc': np.array([0, 0, 1, 0]),
    #                                    'sy': np.array([0, 0, 0, 1])},
    #                          chunk_size=chunk_size,
    #                          tech_density=1500,
    #                          bio_density=bio_density,
    #                          density_mode=density_mode,
    #                          max_step_size=10000000,
    #                          hybrid_mode=False,
    #                          label_mappings=[(3, 1), (4, 1), (5, 0), (6, 0)],
    #                          scheduler='steplr',
    #                          optimizer='adam')
    # training_thread(argscont)

    # 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6
    # today = date.today().strftime("%Y_%m_%d")
    # density_mode = True
    # bio_density = 80
    # sample_num = 60000
    # chunk_size = 30000
    # if density_mode:
    #     name = today + '_{}'.format(bio_density) + '_{}'.format(sample_num)
    # else:
    #     name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)
    # normalization = 50000
    # argscont = ArgsContainer(save_root='/u/jklimesch/thesis/results/param_search_density/full/',
    #                          train_path='/u/jklimesch/thesis/gt/20_02_20/poisson_val_my/',
    #                          val_path='/u/jklimesch/thesis/gt/20_02_20/poisson_val/validation/',
    #                          sample_num=sample_num,
    #                          name=name + f'',
    #                          class_num=7,
    #                          train_transforms=[clouds.RandomVariation((-10, 10)), clouds.RandomRotate(),
    #                                            clouds.Normalization(normalization), clouds.Center()],
    #                          batch_size=2,
    #                          input_channels=1,
    #                          use_val=False,
    #                          val_freq=10,
    #                          features={'hc': 1},
    #                          chunk_size=chunk_size,
    #                          tech_density=1500,
    #                          bio_density=bio_density,
    #                          density_mode=density_mode,
    #                          max_step_size=10000000,
    #                          hybrid_mode=True,
    #                          scheduler='steplr',
    #                          optimizer='adam')
    # training_thread(argscont)

    # 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6
    # today = date.today().strftime("%Y_%m_%d")
    # density_mode = False
    # bio_density = 80
    # sample_num = 28000
    # chunk_size = 10000
    # if density_mode:
    #     name = today + '_{}'.format(bio_density) + '_{}'.format(sample_num)
    # else:
    #     name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)
    # normalization = chunk_size
    # argscont = ArgsContainer(save_root='/u/jklimesch/thesis/results/param_search_context/run4/',
    #                          train_path='/u/jklimesch/thesis/gt/20_02_20/poisson_val_my/',
    #                          val_path='/u/jklimesch/thesis/gt/20_02_20/poisson_val/validation/',
    #                          sample_num=sample_num,
    #                          name=name + f'_co',
    #                          class_num=3,
    #                          train_transforms=[clouds.RandomVariation((-10, 10)), clouds.RandomRotate(),
    #                                            clouds.Normalization(normalization), clouds.Center()],
    #                          batch_size=8,
    #                          input_channels=4,
    #                          use_val=False,
    #                          val_freq=10,
    #                          features={'hc': np.array([1, 0, 0, 0]),
    #                                    'mi': np.array([0, 1, 0, 0]),
    #                                    'vc': np.array([0, 0, 1, 0]),
    #                                    'sy': np.array([0, 0, 0, 1])},
    #                          chunk_size=chunk_size,
    #                          tech_density=1500,
    #                          bio_density=bio_density,
    #                          density_mode=density_mode,
    #                          max_step_size=10000000,
    #                          hybrid_mode=False,
    #                          label_mappings=[(3, 1), (4, 1), (5, 0), (6, 0)],
    #                          scheduler='steplr',
    #                          optimizer='adam')
    # training_thread(argscont)
