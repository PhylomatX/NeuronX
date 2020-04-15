import pickle
import numpy as np
from typing import Tuple, Dict, Any
from morphx.data.basics import load_pkl


class DataContainer(object):
    def __init__(self, density_data: Dict[Any, Tuple[list, list]],
                 context_data: Dict[Any, Tuple[list, list]], time_data: Dict[Any, Tuple[list, list]],
                 metric: str = 'f1-score'):
        self._density_data = density_data
        self._context_data = context_data
        self._time_data = time_data
        self._metric = metric

    @property
    def density_data(self):
        return self._density_data

    @property
    def context_data(self):
        return self._context_data

    @property
    def time_data(self):
        return self._time_data

    @property
    def metric(self):
        return self._metric

    def save2pkl(self, path: str):
        try:
            attr_dict = {'density_data': self.density_data,
                         'context_data': self.context_data,
                         'time_data': self.time_data,
                         'metric': self._metric}
            with open(path, 'wb') as f:
                pickle.dump(attr_dict, f)
            f.close()
        except FileNotFoundError:
            print("Saving was not successful as given path is not valid.")
            return 1
        return 0

    def load_from_pkl(self, path: str):
        self.__init__(**load_pkl(path))
        return self
