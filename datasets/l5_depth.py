###################################################################################################
#
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
YouTube Faces Dataset
https://www.cs.tau.ac.il/~wolf/ytfaces/
"""
import os
import pickle
import time

import numpy
import torch
from torch.utils import data


class L5DepthDataset(data.Dataset):
    """
    Linnaeus 5 128x128 Dataset with MidasNet Depth Output
    """
  
    L5 = {
        'mean': 223.74481,
        'std': 226.20352,
        'max': 809.27625,
        'min': -7.033435,
        'norm_max': 2.588516,
        'norm_min': -1.020224
    }


    def __init__(
            self,
            root_dir,
            d_type,
            transform=None,
    ):
        data_folder = os.path.join(root_dir, d_type)
        assert os.path.isdir(data_folder), (f'No dataset at {data_folder}.'
                                            ' Run the scripts in datasets/depthnet')

        self.sid_list = []
        self.embedding_list = []
        self.img_list = []
        self.transform = transform

        t_start = time.time()
        print('Data loading...')

        # Reading from pickle
        with open(os.path.join(data_folder, d_type+'.pickle'), 'rb') as handle:
            data_pickle = pickle.load(handle)

        self.img_list = data_pickle['img']
        self.embedding_list = data_pickle['embed']
        n_elems = len(self.img_list)
        self.sid_list = list(range(n_elems))
    
        t_end = time.time()
        print(f'{n_elems} of data samples loaded in {t_end-t_start:.4f} seconds.')

    def __normalize_data(self, data_item):  # pylint: disable=no-self-use
        data_item = data_item.astype(numpy.float32)
        data_item /= 256
        return data_item
    
    def __normalize_out(self, data_item):
        std_item = (data_item - self.L5['mean'])/self.L5['std']
        norm_item = numpy.round(255*(std_item - self.L5['norm_min'])/(self.L5['norm_max'] - self.L5['norm_min']))/255
        return norm_item

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        inp = torch.Tensor(self.__normalize_data(self.img_list[idx]))
        inp = inp.permute(2, 0, 1)
        if self.transform is not None:
            inp = self.transform(inp)
        
        embedding = torch.Tensor(self.__normalize_out(self.embedding_list[idx]))

        return inp, embedding
