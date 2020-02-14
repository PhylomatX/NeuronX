import os
import glob

import morphx.processing.objects
from morphx.processing import ensembles, clouds
from morphx.classes.hybridcloud import HybridCloud


def create_ads(input_path: str, output_path: str):
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        hc = morphx.processing.objects.load_pkl(file)
        hc = clouds.map_labels(hc, ['bouton', 'terminal'], 'axon')
        hc = clouds.map_labels(hc, ['neck', 'head'], 'dendrite')
        morphx.processing.objects.save2pkl(hc, output_path, name=name)


if __name__ == '__main__':
    create_ads('/u/jklimesch/gt/gt_poisson/', '/u/jklimesch/gt/gt_poisson/ads/')
