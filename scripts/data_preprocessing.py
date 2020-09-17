from morphx.preprocessing.mesh2poisson import poissonize_dataset
from morphx.preprocessing.voxel_down import voxel_down_dataset

if __name__ == '__main__':
    voxel_down_dataset(f'/u/jklimesch/thesis/gt/cmn/ads/test/raw/',
                       f'/u/jklimesch/thesis/gt/cmn/ads/test/voxeled/', dict(hc=80, mi=100, vc=100, sy=100))

    # poissonize_dataset(f'/u/jklimesch/thesis/tmp/evaluation/batch2/',
    #                    f'/u/jklimesch/thesis/tmp/evaluation/', 100, 0.8)
