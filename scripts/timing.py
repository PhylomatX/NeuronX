import os
import time
import torch
import gpustat
import glob
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from math import ceil
import numpy as np
from morphx.data import basics
from morphx.processing import ensembles, objects, clouds
from elektronn3.models.convpoint import SegBig
from morphx.data.torchhandler import TorchHandler


@torch.no_grad()
def time_convpoint_points(out_path: str, input_channels: int = 4, nclasses: int = 3):
    out_path = os.path.expanduser(out_path)
    batch_points = list(range(7000, 347000, 20000))
    times = []

    for points in tqdm(batch_points):
        device = torch.device('cuda')
        model = SegBig(input_channels, nclasses)
        model.to(device)
        # this function was found by trial and error (finding CUDA memory bounds for different point numbers and
        # fitting the resulting curve)
        batch_size = int(ceil(850000 / points))
        processed = 0
        total_time = 0
        while processed < points * 10:
            pts = torch.rand(batch_size, points, 3)
            feats = torch.ones(batch_size, points, input_channels)
            pts = pts.to(device)
            feats = feats.to(device)
            single_times = []
            for i in range(5):
                start = time.time()
                _ = model(feats, pts)
                single_times.append(time.time() - start)
            total_time += np.mean(single_times)
            processed += batch_size * points
        times.append(total_time/processed*1e6)
        print(f'{points} points with batch size of {batch_size} needed {total_time/processed*1e6} \u03BCs per point')
        print(gpustat.new_query())
        # free CUDA memory
        del model
        torch.cuda.empty_cache()

    plt.scatter(batch_points, times, marker='o', c='b')
    plt.xlabel("points per sample")
    plt.ylabel("time per point in \u03BCs")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path + f'ConvPoint_c{input_channels}_cl{nclasses}.png')
    with open(out_path + f'ConvPoint_c{input_channels}_cl{nclasses}.pkl', 'wb') as f:
        pickle.dump((batch_points, times), f)


@torch.no_grad()
def time_convpoint_batches(out_path: str, input_channels: int = 4, nclasses: int = 7, points: int = 32768):
    out_path = os.path.expanduser(out_path)
    times = []
    batch_sizes = [32, 16, 8, 4, 2]
    for batch_size in batch_sizes:
        device = torch.device('cuda')
        model = SegBig(input_channels, nclasses)
        model.to(device)
        processed = 0
        total_time = 0
        while processed < points * 128:
            pts = torch.rand(batch_size, points, 3)
            feats = torch.ones(batch_size, points, input_channels)
            pts = pts.to(device)
            feats = feats.to(device)
            start = time.time()
            _ = model(feats, pts)
            total_time += time.time() - start
            processed += batch_size * points
        times.append(total_time/processed*1e6)
        print(f'{points} points with batch size of {batch_size} needed {total_time/processed*1e6} \u03BCs per point')
        print(gpustat.new_query())
        # free CUDA memory
        del model
        torch.cuda.empty_cache()

    plt.scatter(batch_sizes, times, marker='o', c='b')
    plt.xlabel("batch sizes")
    plt.ylabel("time per point in \u03BCs")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path + f'ConvPoint_c{input_channels}_cl{nclasses}_batch_p{points}.png')
    with open(out_path + f'ConvPoint_c{input_channels}_cl{nclasses}_batch_p{points}.pkl', 'wb') as f:
        pickle.dump((batch_sizes, times), f)


def evaluate_convpoint_data(file_path: str, out_path: str, points: bool = True):
    """ ConvPoint data: [points / batches, timings] """
    file_path = os.path.expanduser(file_path)
    out_path = os.path.expanduser(out_path)
    data = basics.load_pkl(file_path)
    if points:
        plt.scatter(np.array(data[0])/1e3, np.array(data[1]), c='b', marker='o')
        plt.xlabel('points per sample in 10³')
    else:
        plt.scatter(np.array(data[0]), np.array(data[1]), c='b', marker='o')
        plt.xlabel('batch number')
    plt.ylabel('time per point in \u03BCs')
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def time_splitting(data_path: str, out_path: str, bio_density: float = None, capacity: int = None,
                   tech_density: int = None, density_mode: bool = True, chunk_size: int = None):
    data_path = os.path.expanduser(data_path)
    out_path = os.path.expanduser(out_path)
    if density_mode:
        # calculate number of vertices for extracting max surface area
        vert_num = int(capacity * tech_density / bio_density)
    else:
        vert_num = 0
    files = glob.glob(data_path + '*.pkl')
    timing_info = {}
    for file in tqdm(files):
        print(f"Processing {file}")
        timing_info[file] = {}
        obj = ensembles.ensemble_from_pkl(file)
        timing_info[file]['nodes'] = len(obj.nodes)
        timing_info[file]['vertices'] = len(obj.flattened.vertices)
        timing_info[file]['timing'] = []
        for i in tqdm(range(2)):
            start = time.time()
            base_points = []
            chunks = []
            nodes_new = np.arange(len(obj.nodes))
            mask = np.ones(len(obj.nodes), dtype=bool)
            # choose random node, extract local context at this point, remove local context nodes, repeat until empty
            while len(nodes_new) != 0:
                choice = np.random.choice(nodes_new, 1)
                base_points.append(choice[0])
                if density_mode:
                    if bio_density is None or tech_density is None or capacity is None:
                        raise ValueError('bio_density, tech_density and capacity must be given in density mode')
                    bfs = objects.bfs_vertices_diameter(obj, choice[0], vert_num)
                else:
                    if chunk_size is None:
                        raise ValueError('chunk_size parameter must be given in context mode.')
                    bfs = objects.bfs_euclid_diameter(obj, choice[0], chunk_size)
                chunks.append(bfs)
                mask[bfs] = False
                nodes_new = np.arange(len(obj.nodes))[mask]
            timing_info[file]['timing'].append(time.time() - start)
    if density_mode:
        filename = f'density_d{bio_density}_c{capacity}.pkl'
    else:
        filename = f'context_c{chunk_size}.pkl'
    with open(out_path + filename, 'wb') as f:
        pickle.dump(timing_info, f)


def create_splitting_diagram(data_path: str, out_path: str, threshold: int):
    """ Splitting data: dict, 1. key: sso file, 2. keys: nodes, vertices, timing (multiple values).
        threshold is for removing outlier. """
    data_path = os.path.expanduser(data_path)
    out_path = os.path.expanduser(out_path)
    timing = basics.load_pkl(data_path)
    nodes = []
    times = []
    for key in timing:
        curr = timing[key]
        if curr['nodes'] > threshold:
            continue
        nodes.append(curr['nodes'])
        times.append(np.mean(curr['timing']))
    slashs = [pos for pos, char in enumerate(data_path) if char == '/']
    name = data_path[slashs[-1] + 1:-4]
    plt.scatter(nodes, times, marker='o', c='b')
    plt.xlabel("number of skeleton nodes")
    plt.ylabel("splitting time in s")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path + f'{name}.svg')
    plt.close()


def time_dataloader(data_path: str, sample_num: int = 28000, class_num: int = 7,
                    density_mode: bool = True, chunk_size: int = None, bio_density: int = None):
    data_path = os.path.expanduser(data_path)
    transforms = [clouds.Normalization(50000), clouds.Center()]
    transforms = clouds.Compose(transforms)
    ds = TorchHandler(data_path, sample_num, class_num,
                      density_mode=density_mode,
                      chunk_size=chunk_size,
                      bio_density=bio_density,
                      tech_density=1500,
                      transform=transforms,
                      obj_feats={'hc': np.array([1, 0, 0, 0]), 'mi': np.array([0, 1, 0, 0]),
                                 'vc': np.array([0, 0, 1, 0]), 'sy': np.array([0, 0, 0, 1])},
                      label_mappings=None,
                      specific=True)
    total_time = 0
    total_length = 0
    for i in range(5):
        for obj in tqdm(ds.obj_names):
            total_length += ds.get_obj_length(obj)
            for i in range(ds.get_obj_length(obj)):
                start = time.time()
                sample = ds[(obj, i)]
                total_time += time.time() - start
    print(total_time/total_length)

    ## time dataloader in non specific mode
    # total_time = 0
    # for i in range(5):
    #     for j in tqdm(range(len(ds))):
    #         start = time.time()
    #         sample = ds[j]
    #         total_time += time.time() - start
    # print(total_time/(5*len(ds)))


def evaluate_val_timings(set_path: str, out_path: str, threshold: int = None):
    out_path = os.path.expanduser(out_path)
    set_path = os.path.expanduser(set_path)
    set_dirs = os.listdir(set_path)
    val_data = {}
    eval_data = {}
    for set_di in set_dirs:
        if 'eval' in set_di:
            dirs = os.listdir(set_path + set_di + '/')
            val_vertex_nums = None
            eval_vertex_nums = None
            total_coverage = None
            val_timing = None
            eval_timing = None
            val_counter = 1
            eval_counter = 1
            for di in dirs:
                curr_file = set_path + set_di + '/' + di
                if not os.path.isdir(curr_file):
                    continue
                obj_info = basics.load_pkl(curr_file + '/obj_info.pkl')
                eval_timing_mv = basics.load_pkl(curr_file + '/eval_timing_mv/eval_timing_mv.pkl')
                val_vertices = []
                eval_vertices = []
                val_timings = []
                eval_timings = []
                coverage = np.array([0, 0])
                for key in obj_info:
                    val_vertices.append(obj_info[key]['vertex_num'])
                    val_timings.append(obj_info[key]['timing'])
                for key in eval_timing_mv:
                    if key != 'total':
                        eval_vertices.append(obj_info[key[:-6]]['vertex_num'])
                        eval_timings.append(eval_timing_mv[key]['mv_timing'])
                        coverage += np.array(list(eval_timing_mv[key]['cov']))
                coverage = round((1 - coverage[0] / coverage[1]) * 100)
                if val_vertex_nums is None:
                    val_vertex_nums = np.array(val_vertices)
                else:
                    assert np.all(val_vertices == val_vertex_nums)
                if val_timing is None:
                    val_timing = np.array(val_timings)
                else:
                    assert len(val_timings) == len(val_timing)
                    val_timing += np.array(val_timings)
                    val_counter += 1
                if eval_vertex_nums is None:
                    eval_vertex_nums = np.array(eval_vertices)
                else:
                    assert np.all(eval_vertices == eval_vertex_nums)
                if eval_timing is None:
                    eval_timing = np.array(eval_timings)
                else:
                    assert len(eval_timings) == len(eval_timing)
                    eval_timing += np.array(eval_timings)
                    eval_counter += 1

                if total_coverage is None:
                    total_coverage = coverage
                else:
                    assert total_coverage == coverage
            val_timing /= val_counter
            eval_timing /= eval_counter
            if threshold is not None:
                val_mask = val_vertex_nums/1e6 < threshold
                eval_mask = eval_vertex_nums / 1e6 < threshold
                val_data[total_coverage] = [val_vertex_nums[val_mask], val_timing[val_mask]]
                eval_data[total_coverage] = [eval_vertex_nums[eval_mask], eval_timing[eval_mask]]
            else:
                val_data[total_coverage] = [val_vertex_nums, val_timing]
                eval_data[total_coverage] = [eval_vertex_nums, eval_timing]
    markers = ['o', 'x', '+', 'S', '^', 'H']
    colors = ['b', 'k', 'g', 'r', 'c', 'y', 'm']
    for ix, key in enumerate(val_data):
        plt.scatter(val_data[key][0]/1e6, val_data[key][1], c=colors[ix], marker=markers[ix], label=f'coverage: {key}')
    plt.legend(loc=0)
    plt.xlabel('number of vertices in 10⁶')
    plt.ylabel('timing in s')
    plt.tight_layout()
    plt.grid()
    plt.savefig(out_path + 'val_timing.svg')
    plt.close()

    for ix, key in enumerate(eval_data):
        plt.scatter(eval_data[key][0]/1e6, eval_data[key][1], c=colors[ix],
                    marker=markers[ix], label=f'coverage: {key}')
    plt.legend(loc=0)
    plt.xlabel('number of vertices in 10⁶')
    plt.ylabel('timing in s')
    plt.tight_layout()
    plt.grid()
    plt.savefig(out_path + 'eval_timing.svg')
    plt.close()


if __name__ == '__main__':
    print("")
    # time_splitting('~/thesis/gt/20_02_20/poisson_verts2node/', '~/thesis/results/timings/splitting/',
    #                density_mode=False, chunk_size=20000)
    # time_dataloader('~/thesis/gt/20_02_20/poisson_val/', bio_density=50)

