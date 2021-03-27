# do not remove open3d as import order of open3d and torch is important ###
import open3d as o3d
import torch
import random
import numpy as np
from torch import nn
from datetime import date
import morphx.processing.clouds as clouds
from neuronx.classes.torchhandler import TorchHandler
import elektronn3
elektronn3.select_mpl_backend('Agg')
from lightconvpoint.utils.network import get_search, get_conv
from elektronn3.models.lcp_adapt import ConvAdaptSeg
from elektronn3.training import Trainer3d, Backup
from neuronx.classes.argscontainer import ArgsContainer
from elektronn3.training.schedulers import CosineAnnealingWarmRestarts


def training_thread(acont: ArgsContainer):
    torch.cuda.empty_cache()
    lr = 1e-3
    lr_stepsize = 10000
    lr_dec = 0.995
    max_steps = int(acont.max_step_size / acont.batch_size)

    torch.manual_seed(acont.random_seed)
    np.random.seed(acont.random_seed)
    random.seed(acont.random_seed)

    if acont.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    kwargs = {}
    if acont.model == 'ConvAdaptSeg':
        kwargs = dict(kernel_num=acont.pl, architecture=acont.architecture, activation=acont.act, norm=acont.norm_type)
    conv = dict(layer=acont.conv[0], kernel_separation=acont.conv[1])
    model = ConvAdaptSeg(acont.input_channels, acont.class_num, get_conv(conv), get_search(acont.search), **kwargs)

    batch_size = acont.batch_size

    train_transforms = clouds.Compose(acont.train_transforms)
    train_ds = TorchHandler(data_path=acont.train_path, sample_num=acont.sample_num, nclasses=acont.class_num,
                            feat_dim=acont.input_channels, density_mode=acont.density_mode,
                            ctx_size=acont.chunk_size, bio_density=acont.bio_density,
                            tech_density=acont.tech_density, transform=train_transforms,
                            obj_feats=acont.features, label_mappings=acont.label_mappings,
                            hybrid_mode=acont.hybrid_mode, splitting_redundancy=acont.splitting_redundancy,
                            label_remove=acont.label_remove, sampling=acont.sampling, padding=acont.padding,
                            split_on_demand=acont.split_on_demand, split_jitter=acont.split_jitter,
                            epoch_size=acont.epoch_size, workers=acont.workers, voxel_sizes=acont.voxel_sizes,
                            ssd_exclude=acont.ssd_exclude, ssd_include=acont.ssd_include,
                            ssd_labels=acont.ssd_labels, exclude_borders=acont.exclude_borders,
                            rebalance=acont.rebalance, extend_no_pred=acont.extend_no_pred)

    if acont.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif acont.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.5e-5)
    else:
        raise ValueError('Unknown optimizer')

    if acont.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    elif acont.scheduler == 'cosannwarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)
    else:
        raise ValueError('Unknown scheduler')

    # calculate class weights if necessary
    weights = None
    if acont.class_weights is not None:
        weights = torch.from_numpy(acont.class_weights).float()

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    if acont.use_cuda:
        criterion.cuda()

    if acont.use_val:
        val_path = acont.val_path
    else:
        val_path = None

    trainer = Trainer3d(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_ds,
        v_path=val_path,
        val_freq=acont.val_freq,
        val_red=acont.val_iter,
        channel_num=acont.input_channels,
        batchsize=batch_size,
        num_workers=4,
        save_root=acont.save_root,
        exp_name=acont.name,
        num_classes=acont.class_num,
        schedulers={"lr": scheduler},
        batch_avg=acont.batch_avg,
        target_names=acont.target_names,
        stop_epoch=acont.stop_epoch,
        enable_tensorboard=False
    )
    # Archiving training script, src folder, env info
    Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()
    acont.save2pkl(trainer.save_path + '/argscont.pkl')
    with open(trainer.save_path + '/argscont.txt', 'w') as f:
        f.write(str(acont.attr_dict))
    f.close()

    trainer.run(max_steps)


if __name__ == '__main__':
    # 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6
    today = date.today().strftime("%Y_%m_%d")
    sample_num = 8000
    chunk_size = 8192
    name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)

    # --- Training with cell organelles ---
    features = {'hc': np.array([1, 0, 0, 0]),
                'mi': np.array([0, 1, 0, 0]),
                'vc': np.array([0, 0, 1, 0]),
                'sy': np.array([0, 0, 0, 1])}
    input_channels = 4

    # --- Training with cell organelles and myelin ---
    # features = {'hc': {0: np.array([1, 0, 0, 0, 0]), 1: np.array([0, 1, 0, 0, 0])},
    #             'mi': np.array([0, 0, 1, 0, 0]),
    #             'vc': np.array([0, 0, 0, 1, 0]),
    #             'sy': np.array([0, 0, 0, 0, 1])}
    # input_channels = 5

    # --- Training without organelles ---
    # features = {'hc': np.array([1])}
    # input_channels = 1

    argscont = ArgsContainer(save_root='/u/jklimesch/working_dir/',
                             train_path='/u/jklimesch/working_dir/gt/20_09_27/voxeled/small/',
                             sample_num=sample_num,
                             name=name + f'_test',
                             random_seed=1,
                             class_num=7,
                             train_transforms=[clouds.RandomVariation((-40, 40)),
                                               clouds.RandomRotate(apply_flip=True),
                                               clouds.Center(),
                                               clouds.ElasticTransform(res=(40, 40, 40), sigma=(6, 6)),
                                               clouds.RandomScale(distr_scale=0.1, distr='uniform'),
                                               clouds.Center()],
                             batch_size=16,
                             input_channels=input_channels,
                             use_val=True,
                             val_path='/u/jklimesch/working_dir/gt/20_09_27/voxeled/small/',
                             val_freq=5,
                             features=features,
                             chunk_size=chunk_size,
                             max_step_size=100000000,
                             splitting_redundancy=1,
                             norm_type='gn',
                             label_remove=[-2],
                             label_mappings=[],
                             pl=32,
                             val_label_mappings=[],
                             val_label_remove=[-2],
                             target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head'],
                             model='ConvAdaptSeg',
                             search='SearchFPS')
    training_thread(argscont)
