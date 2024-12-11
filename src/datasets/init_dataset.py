from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .sample.sampler import Sampler
from .sample.sample_new import Sampler

from .dataset.ucf101 import UCF101
from .dataset.hmdb import JHMDB
from .dataset.newbasketball import  NewBasketball


switch_dataset = {
    'ucf101': UCF101,
    'hmdb': JHMDB,
    'newbasketball':NewBasketball
}


def get_dataset(dataset):
    class Dataset(switch_dataset[dataset], Sampler):
        pass
    return Dataset
