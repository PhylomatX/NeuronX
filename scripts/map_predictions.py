from morphx.data.cloudset import CloudSet
from morphx.processing import clouds

if __name__ == '__main__':
    data_path = '/u/jklimesch/thesis/gt/gt_poisson/single/'
    radius = 20000
    npoints = 10000
    transforms = [clouds.RandomVariation((-10, 10)),
                  clouds.Normalization(radius),
                  clouds.RandomRotate(),
                  clouds.Center()]
    transform = clouds.Compose(transforms)
    cs = CloudSet(data_path, radius, npoints, transform, validation=True)
    labels = []
    for ix in range(len(cs)):
        sample = cs[0]
        labels.append(sample.labels)
    for ix in range(len(cs)):
        cs.map_prediction(labels[ix])
