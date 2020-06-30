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
        model = SegBig(acont.input_channels, acont.class_num, trs=acont.track_running_stats, dropout=0,
                       use_bias=acont.use_bias, norm_type=acont.norm_type, use_norm=acont.use_norm,
                       kernel_size=acont.kernel_size, neighbor_nums=acont.neighbor_nums, dilations=acont.dilations,
                       reductions=acont.reductions, first_layer=acont.first_layer, padding=acont.padding)
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
                            hybrid_mode=acont.hybrid_mode,
                            splitting_redundancy=acont.splitting_redundancy,
                            label_remove=acont.label_remove,
                            sampling=acont.sampling,
                            padding=acont.padding)
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
                              hybrid_mode=acont.hybrid_mode,
                              splitting_redundancy=acont.splitting_redundancy,
                              label_remove=acont.label_remove,
                              sampling=acont.sampling)

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

    # calculate class weights if necessary
    weights = None
    if acont.class_weights is not None:
        if isinstance(acont.class_weights, str):
            set_info = train_ds.get_set_info()
            labels = set_info['labels'][0]
            counts = set_info['labels'][1]
            if not acont.hybrid_mode:
                # remove cell organelles
                idcs = labels.argsort()
                counts = counts[idcs][:-(len(labels)-acont.class_num)]
            if acont.class_weights == 'mean':
                weights = counts.mean() / counts
            if acont.class_weights == 'sum':
                weights = counts.sum() / counts
        elif isinstance(acont.class_weights, np.ndarray):
            weights = acont.class_weights
        weights = torch.from_numpy(weights).float()
    print(f"Weights: {weights}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
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
        enable_save_trace=jit,
        collate_fn=None,
        batch_avg=acont.batch_avg
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
    bio_density = 100
    sample_num = 6000
    chunk_size = 10000
    if density_mode:
        name = today + '_{}'.format(bio_density) + '_{}'.format(sample_num)
    else:
        name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)
    if density_mode:
        normalization = 5000
    else:
        normalization = chunk_size

    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}

    argscont = ArgsContainer(save_root='/u/jklimesch/thesis/current_work/model_tests/',
                             train_path='/u/jklimesch/thesis/gt/20_06_09/voxeled/',
                             sample_num=sample_num,
                             name=name + f'',
                             class_num=3,
                             train_transforms=[clouds.RandomVariation((-30, 30)),
                                               clouds.RandomShear(limits=(-0.1, 0.1)),
                                               clouds.RandomRotate(apply_flip=True), clouds.Normalization(normalization),
                                               clouds.Center()],
                             batch_size=16,
                             input_channels=len(features['sy']),
                             use_val=False,
                             features=features,
                             chunk_size=chunk_size,
                             tech_density=100,
                             bio_density=bio_density,
                             density_mode=density_mode,
                             max_step_size=10000000,
                             hybrid_mode=False,
                             label_mappings=[(1, 0), (2, 0), (3, 0), (4, 0), (5, 1), (6, 2)],
                             scheduler='steplr',
                             optimizer='adam',
                             splitting_redundancy=5,
                             use_bias=False,
                             use_norm=True,
                             norm_type='gn',
                             label_remove=[1, 2, 3, 4],
                             sampling=True,
                             kernel_size=16,
                             neighbor_nums=[32, 32, 32, 16, 8, 8, 4, 8, 8, 8, 16, 16, 16],
                             dilations=None,
                             reductions=None,
                             first_layer=True,
                             padding=1000)
    training_thread(argscont)

    # 4-class spine
    # label_mappings = [(2, 1), (3, 1), (4, 1), (5, 2), (6, 3)]
    # 5-class axon
    # label_mappings=[(5, 0), (6, 0)]

    # neighbor_nums = [32, 32, 32, 16, 8, 8, 4, 8, 8, 8, 16, 16, 16]

    # features={'hc': {0: np.array([1, 0, 0, 0, 0]), 1: np.array([0, 1, 0, 0, 0])},
    #           'mi': np.array([0, 0, 1, 0, 0]),
    #           'vc': np.array([0, 0, 0, 1, 0]),
    #           'sy': np.array([0, 0, 0, 0, 1])}
