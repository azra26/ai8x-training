###################################################################################################
#
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to utilize the Linnaeus 5 depth dataset.
"""
import os

from torchvision import transforms

import ai8x
from datasets.l5_depth import L5DepthDataset


def depthnet_get_datasets(data, load_train=True, load_test=True):
    """
    Loads the Linnaeus 5 128x128 Depth Dataset.
    Images and embeddings are loaded from a pickle file
    The images are all 3-color 128x128 sized.
    """
    (data_dir, args) = data

    # These are hard coded for now, need to come from above in future.
    train_test_data_dir = os.path.join(data_dir, 'L5_DEPTH')

    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = L5DepthDataset(root_dir=train_test_data_dir, d_type='train',
                                        transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = L5DepthDataset(root_dir=train_test_data_dir, d_type='test',
                                           transform=transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'DepthNet',
        'input': (3, 128, 128),
        'output': ('embed'),
        'regression': True,
        'loader': depthnet_get_datasets,
    },
]
