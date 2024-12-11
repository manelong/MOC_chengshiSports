from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .base_dataset import BaseDataset


class NewBasketball(BaseDataset):
    num_classes = 1

    def __init__(self, opt, mode):
        assert opt.split == 1, "We use only the first split of NewBasketball"
        self.ROOT_DATASET_PATH = os.path.join(opt.root_dir,'data/newbasketball_v1')
        pkl_filename = 'label_v1_v2_360_640.pkl'
        super(NewBasketball, self).__init__(opt, mode, self.ROOT_DATASET_PATH, pkl_filename)

    def imagefile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, v, '{:0>5}.jpg'.format(i))

    def flowfile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, v, '{:0>5}.jpg'.format(i))
