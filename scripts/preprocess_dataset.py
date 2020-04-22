from morphx.preprocessing.mesh2poisson import poissonize_dataset
from morphx.preprocessing.voxel_down import voxel_down_dataset

if __name__ == '__main__':
    gt_type = 'test'

    # 50, 80
    voxel_down_dataset(f'/u/jklimesch/thesis/gt/cmn/{gt_type}/raw/',
                       f'/u/jklimesch/thesis/gt/cmn/{gt_type}/voxeled/', 100, 150)

    poissonize_dataset(f'/u/jklimesch/thesis/gt/cmn/{gt_type}/raw/',
                       f'/u/jklimesch/thesis/gt/cmn/{gt_type}/poisson/', 300, 0.7)


